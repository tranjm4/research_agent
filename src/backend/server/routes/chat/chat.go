/*
Chat route (src/backend/server/routes/chat.go)

Handles chat-related functionalities:
- Chat messaging
- Chat room management
- Chat history retrieval
- Document management
*/

package chat

import (
	"bufio"
	"database/sql"
	"encoding/json"
	"log"
	"net/http"

	"time"

	"server/routes/chat/util"

	"github.com/go-chi/chi/v5"
)

func RegisterRoutes(r chi.Router, db *sql.DB) {
	// Define chat-related routes here
	r.Route("/chat", func(r chi.Router) {
		r.Post("/send", postSendMessage(db))
		r.Get("/history/{conversationID}", getChatHistory(db))
		r.Get("/conversations", getConversations(db))
		r.Post("/create", postCreateConversation(db))
	})
}

type Message struct {
	ID        int       `json:"id"`
	Content   string    `json:"content"`
	Sender    string    `json:"sender"` // "user" or "assistant"
	Timestamp time.Time `json:"timestamp"`
}

// postSendMessage handles sending a chat message to the LLM server
// It should receive a single message from the user (with metadata such timestamp, type, etc.)
// and send it to the LLM server for processing.
// The response from the LLM server should be a stream of tokens
// that can be sent back to the client in real-time.
type postMessageBody struct {
	Content        string `json:"content"`        // Content of the message
	ConversationID string `json:"conversationId"` // ID of the conversation to which the message belongs
}

func postSendMessage(db *sql.DB) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		// Verify the user session
		sessionExists, err := util.VerifyUserSession(db, r)
		if err != nil {
			log.Println("Failed to verify user session:", err)
			http.Error(w, "Failed to verify user session", http.StatusInternalServerError)
			return
		}
		if !sessionExists {
			log.Println("Unauthorized access attempt")
			http.Error(w, "Unauthorized", http.StatusUnauthorized)
			return
		}

		// Get the message content from the request
		var message postMessageBody
		if err := json.NewDecoder(r.Body).Decode(&message); err != nil {
			log.Println("Failed to decode message:", err)
			http.Error(w, "Failed to decode message", http.StatusBadRequest)
			return
		}
		if message.Content == "" {
			log.Println("Message content is required")
			http.Error(w, "Message content is required", http.StatusBadRequest)
			return
		}
		log.Println("Received message:", message)
		conversationID := message.ConversationID
		// Validate the conversation ownership by checking if the user ID matches the conversation ID
		validated, err := util.VerifyConversationOwnership(db, r, conversationID)
		if err != nil {
			log.Println("Error verifying conversation ownership:", err)
			http.Error(w, "Failed to verify conversation ownership", http.StatusInternalServerError)
			return
		}
		if !validated {
			log.Println("Unauthorized access attempt to conversation")
			http.Error(w, "Unauthorized", http.StatusUnauthorized)
			return
		}

		// User is validated, proceed to send the message to the LLM server

		// Store the message in the database

		// Prepare the message to be send to the LLM server
		userID, err := util.GetUserIDFromToken(db, r)
		if err != nil {
			log.Println("Failed to get user ID from token:", err)
			http.Error(w, "Failed to get user ID", http.StatusInternalServerError)
			return
		}
		err = util.StoreMessage(db, userID, conversationID, message.Content, "user")
		if err != nil {
			log.Println("Failed to store message in database:", err)
			http.Error(w, "Failed to store message", http.StatusInternalServerError)
			return
		}

		// Send the message content to the LLM server
		response, err := util.SendMessageToLLM(r, message.Content, userID, conversationID)
		if err != nil {
			log.Println("Failed to send message to LLM server:", err)
			http.Error(w, "Failed to send message to LLM server", http.StatusInternalServerError)
			return
		}

		fullResponseText := ""

		defer response.Body.Close() // Ensure the response body is closed after use

		if flusher, ok := w.(http.Flusher); ok {
			scanner := bufio.NewScanner(response.Body)
			for scanner.Scan() {
				line := scanner.Text()
				log.Printf("Received line from agent: %s", line)
				if line == "" {
					continue // Skip empty lines
				}
				// Decode the JSON response from the LLM server
				var llmResponse util.Message
				if err := json.Unmarshal([]byte(line), &llmResponse); err != nil {
					log.Printf("Failed to decode LLM response: %v, raw line: %s", err, line)
					errorMsg := util.Message{
						ID:        userID,
						Content:   "Error decoding response from LLM server",
						Sender:    "error",
						Timestamp: time.Now().Format(time.RFC3339),
					}
					json.NewEncoder(w).Encode(errorMsg)
					flusher.Flush()
					continue
				}

				// Append the content to the full response text
				fullResponseText += llmResponse.Content

				if err := json.NewEncoder(w).Encode(&llmResponse); err != nil {
					log.Println("Failed to encode LLM response:", err)
					return
				}
				flusher.Flush() // Flush the response to the client
			}

			if err := scanner.Err(); err != nil {
				log.Printf("Error reading LLM response: %v", err)
			}
		}

		// Store the assistant's response in the database
		err = util.StoreMessage(db, userID, conversationID, fullResponseText, "assistant")
		if err != nil {
			log.Println("Failed to store assistant message in database:", err)
		}

	}
}

type ChatHistoryResponse struct {
	Messages []util.Message `json:"messages"`
	Message  string         `json:"message"`
	OK       bool           `json:"ok"`
}

func getChatHistory(db *sql.DB) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		// Handle retrieving chat history for a specific conversation

		// Given the conversation ID, we would fetch the messages in the conversation

		// Verify the user session
		sessionExists, err := util.VerifyUserSession(db, r)
		if err != nil {
			http.Error(w, "Failed to verify user session", http.StatusInternalServerError)
			return
		}
		if !sessionExists {
			log.Println("Session does not exist or is invalid")
			http.Error(w, "Unauthorized", http.StatusUnauthorized)
			return
		}

		// Session is valid, but we need to ensure the userID associated with the session
		// matches the conversation they are trying to access.
		// Get user ID from the session token
		conversationID := chi.URLParam(r, "conversationID")
		log.Println("Fetching chat history for conversation ID:", conversationID)
		if conversationID == "" {
			log.Println("Conversation ID is required")
			http.Error(w, "Conversation ID is required", http.StatusBadRequest)
			return
		}

		validated, err := util.VerifyConversationOwnership(db, r, conversationID)
		if err != nil {
			log.Println("Error verifying conversation ownership:", err)
			http.Error(w, "Failed to verify conversation ownership", http.StatusInternalServerError)
			return
		}
		if !validated {
			log.Println("Unauthorized access attempt to conversation")
			http.Error(w, "Unauthorized", http.StatusUnauthorized)
			return
		}
		// If the session is valid, we can proceed to fetch the chat history
		// Query the database to get the chat history for the conversation
		messages, err := util.GetMessagesFromConversationID(db, conversationID)
		if err != nil {
			log.Printf("Error getting messages from conversation ID %s: %v", conversationID, err)
			http.Error(w, "Failed to fetch chat history", http.StatusInternalServerError)
			return
		}

		// Prepare the response
		response := ChatHistoryResponse{
			Messages: messages,
			Message:  "Chat history retrieved successfully",
			OK:       true,
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		// Encode the messages to JSON and write to the response
		if err := json.NewEncoder(w).Encode(response); err != nil {
			http.Error(w, "Failed to encode chat history", http.StatusInternalServerError)
			return
		}
	}

}

type Conversation struct {
	ID          int       `json:"id"`
	Title       string    `json:"title"`        // Title of the conversation
	LastUpdated time.Time `json:"last_updated"` // Last updated timestamp
}

type ConversationResponse struct {
	Conversations []Conversation `json:"conversations"`
	Message       string         `json:"message"`
	OK            bool           `json:"ok"`
}

func getConversations(db *sql.DB) http.HandlerFunc {
	// Handle retrieving conversations for the user
	// This involves querying a "conversations" table in the database

	// We would require to verify the user's session token
	// and fetch the conversations associated with the user ID
	// if util.VerifyUserSession(db, )
	return func(w http.ResponseWriter, r *http.Request) {
		// Verify the user session
		sessionExists, err := util.VerifyUserSession(db, r)
		if err != nil {
			log.Printf("Error verifying user session: %v", err)
			http.Error(w, "Failed to verify user session", http.StatusInternalServerError)
			return
		}
		if !sessionExists {
			log.Println("Unauthorized access attempt")
			http.Error(w, "Unauthorized", http.StatusUnauthorized)
			return
		}

		// Get user ID from the session token
		userID, err := util.GetUserIDFromToken(db, r)
		if err != nil {
			log.Printf("Error getting user ID from token: %v", err)
			http.Error(w, "Failed to get user ID", http.StatusInternalServerError)
			return
		}

		// Get the conversations associated with the user ID
		rows, err := db.Query("SELECT id, title FROM conversations WHERE user_id = $1", userID)
		if err != nil {
			log.Printf("Error fetching conversations: %v", err)
			http.Error(w, "Failed to fetch conversations", http.StatusInternalServerError)
			return
		}
		defer rows.Close()

		// Prepare the response

		// Conversations will be a slice of structs
		var conversations []Conversation
		// Iterate through the rows and append to conversations slice
		for rows.Next() {
			var conversation Conversation
			// Scan the row into the conversation struct
			// Check if there is an error scanning the row
			if err := rows.Scan(&conversation.ID, &conversation.Title); err != nil {
				log.Printf("Error scanning conversation: %v", err)
				http.Error(w, "Failed to scan conversation", http.StatusInternalServerError)
				return
			}

			// Append the conversation to the slice
			conversations = append(conversations, conversation)
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		// Encode the conversations slice to JSON and write to the response
		response := ConversationResponse{
			Conversations: conversations,
			Message:       "Conversations retrieved successfully",
			OK:            true,
		}
		if err := json.NewEncoder(w).Encode(response); err != nil {
			log.Printf("Error encoding conversations: %v", err)
			http.Error(w, "Failed to encode conversations", http.StatusInternalServerError)
		}
	}
}

type CreateConversationResponse struct {
	ID        int       `json:"id"`
	Message   string    `json:"message"`
	CreatedAt time.Time `json:"created_at"`
	OK        bool      `json:"ok"`
}

func postCreateConversation(db *sql.DB) http.HandlerFunc {
	// Handle creating a new chat room
	// This involves inserting a new record into a "rooms" table in the database
	return func(w http.ResponseWriter, r *http.Request) {
		// Verify the user session
		sessionExists, err := util.VerifyUserSession(db, r)
		if err != nil {
			log.Printf("Error verifying user session: %v", err)
			http.Error(w, "Failed to verify user session", http.StatusInternalServerError)
			return
		}
		if !sessionExists {
			log.Println("Unauthorized access attempt")
			http.Error(w, "Unauthorized", http.StatusUnauthorized)
			return
		}

		// Session is valid, so create the room

		// Get the conversation title from the request
		title, err := util.GetConversationTitleFromRequest(r)
		if err != nil {
			log.Printf("Error getting conversation title: %v", err)
			http.Error(w, "Failed to get conversation title", http.StatusBadRequest)
			return
		}
		if title == "" {
			log.Println("Conversation title is required")
			http.Error(w, "Conversation title is required", http.StatusBadRequest)
			return
		}
		// Get user ID from the session token
		userID, err := util.GetUserIDFromToken(db, r)
		if err != nil {
			log.Printf("Error getting user ID from token: %v", err)
			http.Error(w, "Failed to get user ID", http.StatusInternalServerError)
			return
		}
		// Insert the new room into the database
		var roomID int
		err = db.QueryRow(
			"INSERT INTO conversations (user_id, title) VALUES ($1, $2) RETURNING id",
			userID, title,
		).Scan(&roomID)
		if err != nil {
			log.Printf("Error creating conversation: %v", err)
			http.Error(w, "Failed to create conversation", http.StatusInternalServerError)
			return
		}
		// Respond with the ID of the new room
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusCreated)
		response := CreateConversationResponse{
			ID:        roomID,
			Message:   "Room created successfully",
			CreatedAt: time.Now(),
			OK:        true,
		}
		if err := json.NewEncoder(w).Encode(response); err != nil {
			log.Printf("Error encoding response: %v", err)
			http.Error(w, "Failed to encode response", http.StatusInternalServerError)
		}
	}
}
