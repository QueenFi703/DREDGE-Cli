import socket
import ssl
import threading
import time
from types import SimpleNamespace
import pytest
import urllib3
from dredge import public_source_collector as collector


@pytest.fixture
def transport(monkeypatch):
    state = {'status': 200, 'headers': {'Content-Type': 'text/html; charset=utf-8'},
             'body': b'<html><head><script>not evidence</script></head><body>Public source content.</body></html>',
             'pool_args': None, 'request': None, 'closed': False}
    monkeypatch.setattr(collector.socket, 'getaddrinfo', lambda *a, **k: [
        (socket.AF_INET, socket.SOCK_STREAM, 6, '', ('8.8.8.8', 443))])
    class Response:
        def __init__(self):
            self.status=state['status']; self.headers=state['headers']; self.body=state['body']
        def read1(self, count, **kwargs):
            if state.get('error'): raise state['error']
            if state.get('on_read'): state['on_read']()
            value,self.body=self.body[:count],self.body[count:]; return value
        def close(self): state['closed']=True
    class Pool:
        def __init__(self, *args, **kwargs): state['pool_args']=(args,kwargs)
        def urlopen(self, *args, **kwargs): state['request']=(args,kwargs); return Response()
        def close(self): state['pool_closed']=True
    monkeypatch.setattr(collector.urllib3, 'HTTPSConnectionPool', Pool)
    return state


def test_pinned_public_ip_and_verified_hostname_no_auth_redirect_proxy(transport):
    result=collector.collect('mo-dss-child-support')
    args, options=transport['pool_args']
    assert args == ('8.8.8.8',)
    assert options['assert_hostname']==options['server_hostname']=='dss.mo.gov'
    assert options['cert_reqs']==ssl.CERT_REQUIRED and options['retries'] is False
    assert '_proxy' not in options
    args, options=transport['request']
    assert args==('GET','/child-support') and options['redirect'] is False
    assert options['headers']['Host']=='dss.mo.gov'
    assert set(options['headers'])=={'Host','User-Agent','Accept','Accept-Encoding','Cache-Control'}
    assert result['text']=='Public source content.'
    assert len(result['digest'])==64 and result['bytes_observed']==len(transport['body'])
    assert transport['closed'] and transport['pool_closed']


@pytest.mark.parametrize('value',['https://dss.mo.gov/child-support','mo-dss-child-support?case=secret',None,{},'unknown'])
def test_no_arbitrary_urls_or_queries(transport,value):
    with pytest.raises(collector.SourceUnavailable): collector.collect(value)
    assert transport['pool_args'] is None


@pytest.mark.parametrize('ip',['127.0.0.1','10.1.1.1','169.254.169.254','192.168.1.1','::1','fc00::1','::ffff:8.8.8.8','224.0.0.1'])
def test_private_reserved_transition_or_mixed_dns_rejected(monkeypatch,transport,ip):
    monkeypatch.setattr(collector.socket,'getaddrinfo',lambda *a,**k:[
        (socket.AF_INET,socket.SOCK_STREAM,6,'',('8.8.8.8',443)),
        (socket.AF_INET,socket.SOCK_STREAM,6,'',(ip,443))])
    with pytest.raises(collector.SourceUnavailable): collector.collect('mo-dss-child-support')
    assert transport['pool_args'] is None


@pytest.mark.parametrize('status',[301,302,307,304,401,403,404,206,500])
def test_redirect_error_partial_not_snapshots(transport,status):
    transport['status']=status;transport['headers']['Location']='https://127.0.0.1/secret'
    with pytest.raises(collector.SourceUnavailable): collector.collect('mo-dss-child-support')
    assert transport['closed']


@pytest.mark.parametrize('headers',[
    {'Content-Type':'application/pdf'}, {'Content-Type':'text/html','Content-Encoding':'gzip'},
    {'Content-Type':'text/html','Content-Length':'9999999'},
    {'Content-Type':'text/html','Content-Length':'-1'},
    {'Content-Type':'text/html','Content-Length':'1'},
    {'Content-Type':'text/html; charset=latin1'},
])
def test_unsupported_or_incomplete_response_rejected(transport,headers):
    transport['headers']=headers
    with pytest.raises(collector.SourceUnavailable): collector.collect('mo-dss-child-support')


def test_stream_size_timeout_and_invalid_encoding(transport):
    transport['body']=b'x'*(collector.MAX_BYTES+1)
    with pytest.raises(collector.SourceUnavailable): collector.collect('mo-dss-child-support')
    transport['body']=b'\xff'
    with pytest.raises(collector.SourceUnavailable): collector.collect('mo-dss-child-support')
    transport['error']=urllib3.exceptions.ReadTimeoutError(None,'fixture','timeout')
    with pytest.raises(collector.SourceUnavailable): collector.collect('mo-dss-child-support')


def test_eof_crossing_wall_deadline_is_not_an_observation(monkeypatch, transport):
    clock = [0.0]
    monkeypatch.setattr(collector, 'time', SimpleNamespace(monotonic=lambda: clock[0], time=time.time))
    elapsed = iter([11.9, 13.8])
    transport['on_read'] = lambda: clock.__setitem__(0, next(elapsed))
    with pytest.raises(collector.SourceUnavailable, match='deadline'):
        collector.collect('mo-dss-child-support')
    assert transport['closed'] and transport['pool_closed']


def test_wall_timer_interrupts_socket_backed_read_without_network(monkeypatch):
    # Stand in for a peer that never finishes its header/body despite continuing
    # to send bytes. No socket is opened and no network transport is invoked.
    interrupted = threading.Event()
    closed = threading.Event()

    class Socket:
        def shutdown(self, how):
            assert how == socket.SHUT_RDWR
            interrupted.set()
        def close(self):
            closed.set()

    def connect(connection):
        connection.sock = Socket()

    monkeypatch.setattr(urllib3.connection.HTTPSConnection, 'connect', connect)
    timers = []
    connection = collector._DeadlineHTTPSConnection('8.8.8.8', timeout=3,
                                                    deadline=time.monotonic() + .05,
                                                    deadline_timers=timers)
    try:
        connection.connect()
        assert connection.timeout <= .025
        # A Connection: close response detaches its socket after headers, while
        # its response body retains a makefile. That must not cancel the timer.
        connection.close()
        assert interrupted.wait(1), 'Socket-backed read was not interrupted'
        assert closed.wait(1)
    finally:
        connection.close()
        for timer in timers:
            timer.cancel()


def test_collect_cancels_deadline_timers(monkeypatch, transport):
    events = []
    class Timer:
        def cancel(self): events.append('cancel')
    def register_timer():
        timers = transport['pool_args'][1]['deadline_timers']
        if not timers:
            timers.append(Timer())
    transport['on_read'] = register_timer
    collector.collect('mo-dss-child-support')
    assert events == ['cancel']
