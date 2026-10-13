"""Fetch only named public institutional pages. No user URL, proxy, cookie or auth.

DNS answers are checked then pinned to a single public IP with hostname-verified
TLS. Redirects, compression, non-HTML, partial and oversized responses fail closed.
"""
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from html.parser import HTMLParser
import hashlib
import ipaddress
import socket
import ssl
import threading
import time
from urllib.parse import urlsplit

import urllib3

CATALOG = {
    'mo-dss-child-support': ('Missouri DSS: Child Support', 'https://dss.mo.gov/child-support'),
    'mo-dss-custodial': ('Missouri DSS: Custodial Parent', 'https://dss.mo.gov/child-support/custodial-parent'),
    'mo-dss-noncustodial': ('Missouri DSS: Non-Custodial Parent', 'https://dss.mo.gov/child-support/noncustodial-parent'),
    'mo-sos-title13': ('Missouri SOS: Title 13 regulations index', 'https://www.sos.mo.gov/adrules/csr/current/13csr/13csr'),
    'federal-css-policy': ('Federal Child Support: Policy', 'https://acf.gov/css/policy'),
}
VERSION = 'public-html-exact-bytes-v1'
MAX_BYTES = 512 * 1024
MAX_TEXT = 120000
WALL_SECONDS = 12
_RESOLVERS = ThreadPoolExecutor(max_workers=2, thread_name_prefix='public-source-dns')
_DNS_SLOTS = threading.BoundedSemaphore(2)


class SourceUnavailable(ValueError):
    pass


class _DeadlineHTTPSConnection(urllib3.connection.HTTPSConnection):
    """Interrupt header/body reads even when a peer keeps dripping bytes."""
    def __init__(self, *args, deadline, deadline_timers, **kwargs):
        super().__init__(*args, **kwargs)
        self._deadline = deadline
        self._deadline_timers = deadline_timers

    def connect(self):
        remaining = self._deadline - time.monotonic()
        if remaining <= 0:
            raise SourceUnavailable('Public-source deadline exceeded.')
        # TCP connect and TLS handshake each consume a socket timeout. Reserve
        # half the remaining budget for each; neither can run beyond the budget.
        self.timeout = min(self.timeout, remaining / 2)
        super().connect()
        remaining = self._deadline - time.monotonic()
        if remaining <= 0:
            self.close()
            raise SourceUnavailable('Public-source deadline exceeded.')
        sock = self.sock

        def expire():
            # shutdown interrupts a makefile-backed read; close alone does not.
            # Capture this exact socket so a late callback cannot close a new one.
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            finally:
                sock.close()

        timer = threading.Timer(remaining, expire)
        timer.daemon = True
        # Keep the timer until collect finishes: HTTPConnection.close may run
        # after headers while the response still owns a socket-backed file.
        self._deadline_timers.append(timer)
        timer.start()


def catalog_key(url):
    return next((key for key, (_, allowed) in CATALOG.items() if url == allowed), None)


def _resolve(host):
    if not _DNS_SLOTS.acquire(blocking=False):
        raise SourceUnavailable('Public-source DNS is busy; retry later.')
    def query():
        try:
            return socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)
        finally:
            _DNS_SLOTS.release()
    future = _RESOLVERS.submit(query)
    try:
        records = future.result(timeout=3)
    except (TimeoutError, OSError):
        raise SourceUnavailable('Public-source DNS was unavailable.') from None
    addresses = []
    for record in records:
        address = record[4][0]
        try:
            ip = ipaddress.ip_address(address)
        except ValueError:
            raise SourceUnavailable('Invalid public-source DNS answer.') from None
        if not ip.is_global or ip.is_multicast or ip.is_unspecified or ip.is_reserved:
            raise SourceUnavailable('Public-source DNS resolved outside public unicast space.')
        # Reject IPv4-mapped IPv6 and transition ranges; never reinterpret them.
        if isinstance(ip, ipaddress.IPv6Address) and (ip.ipv4_mapped or ip.sixtofour or ip.teredo):
            raise SourceUnavailable('Public-source DNS returned an unsupported transition address.')
        addresses.append(str(ip))
    if not addresses:
        raise SourceUnavailable('Public-source DNS returned no address.')
    return sorted(set(addresses))[0]


class VisibleText(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.hidden = 0
        self.parts = []
    def handle_starttag(self, tag, attrs):
        if tag in {'script', 'style', 'head', 'noscript', 'template'}:
            self.hidden += 1
    def handle_endtag(self, tag):
        if tag in {'script', 'style', 'head', 'noscript', 'template'} and self.hidden:
            self.hidden -= 1
    def handle_data(self, data):
        if not self.hidden:
            self.parts.append(data)


def collect(key):
    """Return an observation only for a complete, successful bounded page response."""
    if not isinstance(key, str) or key not in CATALOG:
        raise SourceUnavailable('Source is not in the approved public catalog.')
    title, url = CATALOG[key]
    parsed = urlsplit(url)
    start = time.monotonic()
    address = _resolve(parsed.hostname)
    remaining = WALL_SECONDS - (time.monotonic() - start)
    if remaining <= 0:
        raise SourceUnavailable('Public-source deadline exceeded.')
    deadline_timers = []
    pool = urllib3.HTTPSConnectionPool(
        address, port=443, server_hostname=parsed.hostname, assert_hostname=parsed.hostname,
        cert_reqs=ssl.CERT_REQUIRED, ssl_minimum_version=ssl.TLSVersion.TLSv1_2,
        timeout=urllib3.Timeout(connect=min(3, remaining), read=min(2, remaining), total=remaining),
        retries=False, maxsize=1, block=True,
        deadline=start + WALL_SECONDS, deadline_timers=deadline_timers,
    )
    pool.ConnectionCls = _DeadlineHTTPSConnection
    response = None
    try:
        response = pool.urlopen('GET', parsed.path, headers={
            'Host': parsed.hostname, 'User-Agent': 'DREDGE-PublicSourceCheck/1.0',
            'Accept': 'text/html', 'Accept-Encoding': 'identity', 'Cache-Control': 'no-cache',
        }, redirect=False, retries=False, preload_content=False, decode_content=False,
            assert_same_host=False, pool_timeout=1)
        if response.status != 200:
            raise SourceUnavailable('Source did not return a complete HTTP 200 page; redirects are not followed.')
        content_type = response.headers.get('Content-Type', '').lower()
        if content_type.split(';', 1)[0].strip() not in {'text/html', 'application/xhtml+xml'}:
            raise SourceUnavailable('Only public HTML pages are supported.')
        if response.headers.get('Content-Encoding', 'identity').lower() not in {'', 'identity'}:
            raise SourceUnavailable('Compressed responses are not accepted.')
        declared = response.headers.get('Content-Length')
        if declared is not None and (not declared.isdecimal() or int(declared) > MAX_BYTES):
            raise SourceUnavailable('Source response size is invalid or exceeds the limit.')
        chunks, length = [], 0
        while True:
            if time.monotonic() - start > WALL_SECONDS:
                raise SourceUnavailable('Public-source deadline exceeded.')
            chunk = response.read1(min(16384, MAX_BYTES + 1 - length), decode_content=False)
            if time.monotonic() - start >= WALL_SECONDS:
                raise SourceUnavailable('Public-source deadline exceeded.')
            if not chunk:
                break
            length += len(chunk)
            if length > MAX_BYTES:
                raise SourceUnavailable('Source response exceeds the byte limit.')
            chunks.append(chunk)
        raw = b''.join(chunks)
        if not raw or (declared is not None and len(raw) != int(declared)):
            raise SourceUnavailable('Source response was empty or incomplete.')
        # Only explicitly UTF-8/ASCII or undeclared HTML encoding is supported.
        if 'charset=' in content_type and content_type.split('charset=', 1)[1].strip(' "\'') not in {'utf-8', 'utf8', 'us-ascii'}:
            raise SourceUnavailable('Source character encoding is unsupported.')
        try:
            html = raw.decode('utf-8')
        except UnicodeDecodeError:
            raise SourceUnavailable('Source was not valid UTF-8 HTML.') from None
        parser = VisibleText()
        parser.feed(html)
        text = ' '.join(' '.join(parser.parts).split())
        if not text or len(text) > MAX_TEXT:
            raise SourceUnavailable('Extracted source text is empty or too large.')
        if time.monotonic() - start >= WALL_SECONDS:
            raise SourceUnavailable('Public-source deadline exceeded.')
        return dict(catalog_id=key, url=url, title=title,
                    digest=hashlib.sha256(raw).hexdigest(), text=text,
                    observed_at=time.time(), collector_version=VERSION, bytes_observed=len(raw))
    except SourceUnavailable:
        raise
    except (urllib3.exceptions.HTTPError, OSError, ValueError):
        raise SourceUnavailable('Public-source HTTPS observation failed; no snapshot was accepted.') from None
    finally:
        for timer in deadline_timers:
            timer.cancel()
        if response is not None:
            response.close()
        pool.close()
