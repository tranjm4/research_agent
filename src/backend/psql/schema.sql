CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    username TEXT NOT NULL UNIQUE,
    email TEXT NOT NULL UNIQUE,
    password_hash VARCHAR(60) NOT NULL,
    created_at TIMESTAMP DEFAULT now()
);

CREATE TABLE IF NOT EXISTS conversations (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    title TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT now(),
    last_modified TIMESTAMP DEFAULT now()
);

CREATE TABLE IF NOT EXISTS messages (
    id SERIAL PRIMARY KEY,
    sender_type TEXT NOT NULL CHECK (sender_type IN ('user', 'assistant')),
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT now(),
    conversation_id INTEGER NOT NULL REFERENCES conversations(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS documents (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    conversation_id INTEGER NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    authors VARCHAR(255)[],
    added_at TIMESTAMP DEFAULT now()
);

CREATE TABLE IF NOT EXISTS session_tokens (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    token TEXT NOT NULL UNIQUE,
    expires_at TIMESTAMP NOT NULL,
    created_at TIMESTAMP DEFAULT now(),
    is_valid BOOLEAN DEFAULT TRUE
);

-- This table is used to log user login attempts, logouts, and registrations
-- It can be useful for auditing and security purposes
CREATE TABLE IF NOT EXISTS login_logs (
    id SERIAL PRIMARY KEY,
    username TEXT NOT NULL,
    activity_type TEXT NOT NULL CHECK (type IN ('login', 'logout', 'register')),
    activity_time TIMESTAMP DEFAULT now(),
    ip_address TEXT NOT NULL,
    success BOOLEAN NOT NULL,
    user_agent TEXT NOT NULL
);

CREATE INDEX idx_conversations_user_id ON conversations (user_id);
CREATE INDEX idx_documents_conversation_id ON documents (conversation_id);
CREATE INDEX idx_documents_user_id ON docuemnts (user_id);
CREATE INDEX idx_session_tokens_user_id ON session_tokens (user_id);
CREATE INDEX idx_documents_user_id ON documents (user_id);
