"""Copyright credits stay visible without changing page access controls."""

from pathlib import Path

import pytest

from test_studio import app, client_for


NOTICE = "© 2026 Sophia Cole (doing business as COLEWORLD)"
COPYRIGHT_URL = "https://github.com/QueenFi703/DREDGE-Cli/blob/main/COPYRIGHT"


def test_railway_image_retains_notices_for_build_and_runtime():
    root = Path(__file__).resolve().parents[1]
    dockerfile = (root / "Dockerfile.railway").read_text()
    assert dockerfile.count("COPY README.md LICENSE COPYRIGHT AUTHORS.md ./") == 2
    dockerignore = (root / ".dockerignore").read_text().splitlines()
    assert "!AUTHORS.md" in dockerignore
    assert "!COPYRIGHT" in dockerignore


@pytest.mark.parametrize("path", ["/auth/login", "/architecture", "/preview", "/pilot-demo", "/billing"])
def test_public_pages_display_copyright_notice(app, path):
    response = app.test_client().get(path)
    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert NOTICE in html
    assert f'href="{COPYRIGHT_URL}"' in html


@pytest.mark.parametrize("path", ["/advanced", "/casework", "/client"])
def test_private_pages_display_copyright_notice_after_sign_in(app, path):
    assert app.test_client().get(path).status_code == 302
    client, _ = client_for(app)
    response = client.get(path)
    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert NOTICE in html
    assert f'href="{COPYRIGHT_URL}"' in html
