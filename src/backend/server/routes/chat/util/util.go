/*
File: src/backend/server/routes/chat/util/util.go

This file contains utility functions for the chat module.
*/

package util

import (
	"database/sql"
	"log"

	"encoding/json"

	"bytes"
	"net/http"
	"os"
	"strconv"
	"time"

	"github.com/joho/godotenv"
)

func GetDBConnection(db *sql.DB) *sql.DB {
	if db == nil {
		log.Fatal("Database connection is nil")
	}
	return db
}

func GetJWTTokenFromRequest(r *http.Request) (string, error) {
	cookie, err := r.Cookie("session_token")
	if err != nil {
		return "", err
	}
	return cookie.Value, nil
}

func VerifyUserSession(db *sql.DB, r *http.Request) (bool, error) {
	if db == nil {
		return false, sql.ErrConnDone
	}

	token, err := GetJWTTokenFromRequest(r)
	if err != nil {
		log.Printf("Error getting JWT token from request: %v", err)
		return false, err
	}
	// Query the database to check if the session token exists
	// and if it is valid
	var exists bool
	err = db.QueryRow("SELECT EXISTS(SELECT 1 FROM session_tokens WHERE token = $1 AND is_valid = true)", token).Scan(&exists)
	if err != nil {
		log.Printf("Error querying session_tokens: %v", err)
		return false, err
	}
	return exists, nil
}

func GetUserIDFromToken(db *sql.DB, r *http.Request) (int, error) {
	// request: application/json
	token, err := GetJWTTokenFromRequest(r)
	if err != nil {
		return 0, err
	}
	// Query the database to get the user ID from the session token
	var userID int
	err = db.QueryRow("SELECT user_id FROM session_tokens WHERE token = $1", token).Scan(&userID)
	if err != nil {
		return 0, err
	}
	return userID, nil
}

type Message struct {
	ID        int    `json:"id"`
	Content   string `json:"content"`
	Timestamp string `json:"timestamp"`
	Sender    string `json:"sender"` // 'user' or 'assistant'
}

func GetMessagesFromConversationID(db *sql.DB, conversationID string) ([]Message, error) {
	conversationIDInt, err := strconv.Atoi(conversationID)
	if err != nil {
		return nil, err
	}
	// Query the database to get messages for the conversation ID
	rows, err := db.Query("SELECT id, content, created_at, sender_type FROM messages WHERE conversation_id = $1", conversationIDInt)
	if err != nil {
		return nil, err
	}
	defer rows.Close() // Close the cursor after use

	var messages []Message
	// Create a slice to hold the messages
	for rows.Next() {
		var msg Message
		// Scan the row into the message struct
		// Check if there is an error scanning the row
		if err := rows.Scan(&msg.ID, &msg.Content, &msg.Timestamp, &msg.Sender); err != nil {
			return nil, err
		}
		messages = append(messages, msg)
	}
	return messages, nil
}

func GetConversationTitleFromRequest(r *http.Request) (string, error) {
	// Given a POST request, extract the conversation title from the request body
	var requestBody struct {
		Title string `json:"title"`
	}
	if err := json.NewDecoder(r.Body).Decode(&requestBody); err != nil {
		return "", err
	}
	if requestBody.Title == "" {
		return "", http.ErrNoCookie
	}
	return requestBody.Title, nil
}

type UserInput struct {
	Message        string `json:"user_input"`
	ConversationID string `json:"conversation_id"`
}

func SendMessageToLLM(r *http.Request, message string, userID int, conversationID string) (*http.Response, error) {
	log.Println("Sending message to LLM server:", message)
	// Given a message, send it to the LLM server and return the response
	if message == "" {
		return nil, http.ErrNoCookie
	}
	// Server for FastAPI
	err := godotenv.Load(".env") // Load environment variables from .env file
	if err != nil {
		log.Println("Error loading .env file:", err)
		return nil, err
	}

	serverURL := os.Getenv("LLM_SERVER_URL")
	if serverURL == "" {
		// Fallback to a default server URL if not set in the environment
		log.Println("LLM_SERVER_URL not set, using default server URL")
		// Default server URL for local development
		serverURL = "http://localhost:8000"
	}

	// Create a new HTTP request
	llmRequest := UserInput{
		Message:        message,
		ConversationID: conversationID,
	}
	llmRequestJSON, err := json.Marshal(llmRequest)
	if err != nil {
		return nil, err
	}

	req, err := http.NewRequest("POST",
		serverURL+"/chat", bytes.NewReader(llmRequestJSON))
	if err != nil {
		return nil, err
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "text/event-stream")
	// Enable CORS for the request
	req.Header.Set("Access-Control-Allow-Origin", "*")

	client := &http.Client{Timeout: 60 * time.Second}
	resp, err := client.Do(req)
	log.Println("Response status from LLM server:", resp.Status)
	if err != nil {
		return nil, err
	}
	return resp, nil
}

// TODO: Fix typing in SQL query
func VerifyConversationOwnership(db *sql.DB, r *http.Request, conversationID string) (bool, error) {
	// Verify if the user owns the conversation
	userID, err := GetUserIDFromToken(db, r)
	if err != nil {
		return false, err
	}
	var ownerID int
	err = db.QueryRow("SELECT user_id FROM conversations WHERE id = $1", conversationID).Scan(&ownerID)
	if err != nil {
		return false, err
	}
	return ownerID == userID, nil
}

func StoreMessage(db *sql.DB, userID int, conversationID string, content string, senderType string) error {
	// Store the message in the database
	_, err := db.Exec("INSERT INTO messages (user_id, conversation_id, content, sender_type) VALUES ($1, $2, $3, $4)",
		userID, conversationID, content, senderType)
	return err
}
