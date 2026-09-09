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
    assert "Ecommerce Agent" in response.text
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
    assert "/sessions" in body
    assert "/?session_id=" in body
    assert 'href="/"' in body
    assert "Ecommerce Agent" in body
    assert "last_question" in body
    assert "turn_count" in body
