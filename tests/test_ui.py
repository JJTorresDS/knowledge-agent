def test_chat_ui(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "<html" in response.text.lower()
    assert "session_id" in response.text
    assert "sessionStorage" in response.text
    assert "/feedback" in response.text
    assert "thumbs-up" in response.text
    assert "thumbs-down" in response.text
    assert "turn_id" in response.text
    assert "Helpful" in response.text
    assert "Not helpful" in response.text
    assert "agent-turn" in response.text
    assert "send(1," in response.text
    assert "send(-1," in response.text
    assert "Feedback failed" in response.text
    assert "Knowledge Agent" in response.text
    assert "Ecommerce Agent" not in response.text
    assert "Local Agent" not in response.text
    assert 'id="session-mode"' in response.text
    assert 'value="auto"' in response.text
    assert 'value="custom"' in response.text
    assert "Auto" in response.text
    assert "Custom" in response.text
    assert 'id="custom-session-id"' in response.text
    assert 'href="/admin"' in response.text
    assert "loadConversation" in response.text
    assert "/sessions/" in response.text
    assert "URLSearchParams" in response.text
    assert "localStorage" in response.text
    assert "setInterval" in response.text
    assert "3000" in response.text
    assert "document.hidden" in response.text
    assert "asking" in response.text
    assert 'id="llm-model"' in response.text
    assert "fetch('/models')" in response.text
    assert "model:" in response.text or '"model"' in response.text


def test_ecommerce_catalog_ui(client):
    response = client.get("/ecommerce")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    body = response.text
    assert "KID LOVE VEST" in body
    assert "G-001" in body


def test_admin_ui(client):
    response = client.get("/admin")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    body = response.text
    assert "<html" in body.lower()
    assert "Admin" in body
    assert "Ongoing conversations" in body
    assert "/admin/conversations" in body
    assert "fetch(" in body
    assert "limit=" in body
    assert "min_turns=" in body
    assert "setInterval" in body
    assert "3000" in body
    assert "document.hidden" in body
    assert "unpkg.com" not in body
    assert "htmx.org" not in body
    assert "hx-get" not in body
    assert 'href="/"' in body
    assert "Knowledge Agent" in body
    assert "Ecommerce Agent" not in body
    assert "Loading conversations" in body
    assert 'id="conversation-limit"' in body
    assert 'id="min-turns"' in body
    assert 'value="10"' in body
