/*
Auth route (src/backend/server/routes/auth.go)

Handles authentication funcationalities:
- User login
- User signup
- User session management
- Password reset
- Token generation and validation
- Middleware for protected routes
- User profile management
- Role-based access control
- JWT token handling
- OAuth integration
- User account verification
- Password hashing and security
- Session expiration and renewal
*/

package userauth

import (
	"database/sql"
	"encoding/json"
	"net/http"

	"log"
	"time"

	"github.com/go-chi/chi/v5"
	"github.com/google/uuid"
	"golang.org/x/crypto/bcrypt"
)

func RegisterRoutes(r chi.Router, db *sql.DB) {
	// Define user authentication-related routes here
	r.Route("/auth", func(r chi.Router) {
		r.Post("/register", postRegister(db))
		r.Post("/login", postLogin(db))
		r.Post("/logout", postLogout(db))
	})

	// Additional routes can be added here as needed
}

type RegisterRequest struct {
	Username string `json:"username"`
	Email    string `json:"email"`
	Password string `json:"password"`
}

type LoginRequest struct {
	Username string `json:"username"`
	Password string `json:"password"`
}

// postLogin handles user login
// It should validate the credentials, generate a session token, and return it to the client.
// It can also set a cookie with the session token for subsequent requests.
// This function should return an appropriate HTTP response indicating success or failure.
// It can also handle errors such as invalid credentials or server issues.
func postLogin(db *sql.DB) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		// handle user login logic
		// Collect request data
		var request LoginRequest
		err := json.NewDecoder(r.Body).Decode(&request)
		if err != nil {
			log.Println("Failed to decode request body (Invalid JSON body):", err)
			http.Error(w, "Invalid JSON request body", http.StatusBadRequest)
			return
		}
		username := request.Username
		password := request.Password
		if username == "" || password == "" {
			log.Println("Username or password is empty")
			http.Error(w, "Username or password is required", http.StatusBadRequest)
			return
		}

		// Validate credentials
		// verify the user exists
		userValidateQuery := `SELECT password_hash FROM users WHERE username = $1 OR email = $1`
		// if query returns no rows, user does not exist
		var hashedPassword string
		err = db.QueryRow(userValidateQuery, username).Scan(&hashedPassword)
		if err == sql.ErrNoRows {
			log.Printf("User %s not found", username)
			http.Error(w, "Invalid username or password", http.StatusUnauthorized)
			return
		}
		err = bcrypt.CompareHashAndPassword([]byte(hashedPassword), []byte(password))
		if err != nil {
			log.Printf("Failed to authenticate user %s: %v", username, err)
			http.Error(w, "Invalid username or password", http.StatusUnauthorized)
			return
		}

		// If we reach this point, the user is authenticated
		// Generate a session token and return it to the client
		sessionToken, err := generateSessionToken(username, db)
		if err != nil {
			log.Printf("Failed to generate session token for user %s: %v", username, err)
			http.Error(w, "Failed to generate session token", http.StatusInternalServerError)
			return
		}

		// Set the session token in a cookie
		http.SetCookie(w, &http.Cookie{
			Name:     "session_token",
			Value:    sessionToken,
			Path:     "/",                            // Set the cookie path to root so it's accessible across the site
			HttpOnly: true,                           // Prevent JavaScript access to the cookie
			Secure:   true,                           // Use secure cookies in production
			Expires:  time.Now().Add(24 * time.Hour), // Set cookie expiration to 24 hours
			SameSite: http.SameSiteLaxMode,           // Use Lax SameSite policy for CSRF protection
		})

		// Return a success response
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(`{"message": "Login successful"}`))
	}
}

// postLogout handles user logout
// It should invalidate the session token, clear the cookie, and return a success response.
// The session token should be removed from the database or marked as invalid.
// This function should return an appropriate HTTP response indicating success or failure.
// It can also handle errors such as server issues or invalid session tokens.
func postLogout(db *sql.DB) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {

		// Clear the session token cookie
		http.SetCookie(w, &http.Cookie{
			Name:  "session_token",
			Value: "",
			Path:  "/",
		})

		// invalidate the session token in the database
		invalidateQuery := `UPDATE session_tokens SET valid = false WHERE token = $1`
		sessionToken, err := r.Cookie("session_token")
		if err != nil {
			http.Error(w, "Session token not found", http.StatusUnauthorized)
			return
		}
		_, err = db.Exec(invalidateQuery, sessionToken.Value)
		if err != nil {
			http.Error(w, "Failed to invalidate session token", http.StatusInternalServerError)
			return
		}
		// Return a success JSON response
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(`{"message": "Logged out successfully"}`))

		// Redirect the user to the home page
		http.Redirect(w, r, "/", http.StatusSeeOther)
	}
}

// postRegister handles user registration
// It should validate the input (e.g., check if the user and email already exists),
// hash the password, and insert the new user into the database.
// It should also handle any errors that may occur during the registration process.
// This function should return an appropriate HTTP response indicating success or failure.
// It can also send a confirmation email if needed.
func postRegister(db *sql.DB) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		// Handle user registration logic
		// This could involve inserting a new user record into the database
		validateQuery := `SELECT COUNT(*) FROM users WHERE username = $1 OR email = $2`
		var count int

		var request RegisterRequest
		err := json.NewDecoder(r.Body).Decode(&request)
		if err != nil {
			log.Println("Failed to decode request body (Invalid JSON body):", err)
			http.Error(w, "Invalid JSON request body", http.StatusBadRequest)
			return
		}
		username := request.Username
		email := request.Email
		password := request.Password

		if username == "" || email == "" || password == "" {
			log.Println("Username, Email, or Password is empty")
			http.Error(w, "Username, email, and password are required", http.StatusBadRequest)
			return
		}

		err = db.QueryRow(validateQuery, username, email).Scan(&count)
		if err != nil {
			log.Printf("Error checking user existence: %v", err)
			http.Error(w, "Internal Server Error", http.StatusInternalServerError)
			return
		}
		if count > 0 {
			log.Printf("User already exists: %s", username)
			// Return a conflict error if the user already exists
			http.Error(w, "User already exists", http.StatusConflict)
			return
		}

		// Hash the password
		hashedPassword, err := bcrypt.GenerateFromPassword([]byte(password), bcrypt.DefaultCost)
		if err != nil {
			log.Printf("Error hashing password: %v", err)
			http.Error(w, "Internal Server Error", http.StatusInternalServerError)
			return
		}

		// Insert the new user into the database
		insertQuery := `
		INSERT INTO users (username, email, password_hash) VALUES ($1, $2, $3)`
		_, err = db.Exec(insertQuery, username, email, hashedPassword)
		if err != nil {
			log.Printf("Error inserting new user: %v", err)
			http.Error(w, "Internal Server Error", http.StatusInternalServerError)
			return
		}

		// Generate a session token for the new user
		sessionToken, err := generateSessionToken(username, db)
		if err != nil {
			log.Printf("Error generating session token: %v", err)
			http.Error(w, "Failed to generate session token", http.StatusInternalServerError)
			return
		}
		// Set the session token in a cookie
		http.SetCookie(w, &http.Cookie{
			Name:     "session_token",
			Value:    sessionToken,
			Path:     "/",                            // Set the cookie path to root so it's accessible across the site
			HttpOnly: true,                           // Prevent JavaScript access to the cookie
			Secure:   true,                           // Use secure cookies in production
			Expires:  time.Now().Add(24 * time.Hour), // Set cookie expiration to 24 hours
			SameSite: http.SameSiteLaxMode,           // Use Lax SameSite policy for CSRF protection
		})

		// Send a confirmation email if needed

		// Return a success response
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusCreated)
		w.Write([]byte(`{"message": "User registered successfully"}`))
	}
}

// generateSessionToken generates a session token for the user
// This function should create a secure token, store it in the database,
// and return it to the client.
func generateSessionToken(username string, db *sql.DB) (string, error) {
	// Create a new token
	token := uuid.New().String()

	// Insert the session token into the database
	err := insertSessionToDB(db, username, token)
	if err != nil {
		return "", err
	}

	// Return the token
	return token, nil
}

func insertSessionToDB(db *sql.DB, username string, token string) error {
	// get the user ID from the users table
	var userID int
	err := db.QueryRow("SELECT id FROM users WHERE username = $1", username).Scan(&userID)
	if err != nil {
		return err
	}
	// Insert the session token into the sessions table with an expiration date
	expiration := time.Now().Add(24 * time.Hour) // Set expiration to 24 hours from now
	_, err = db.Exec(
		`INSERT INTO session_tokens (user_id, token, expires_at) 
		VALUES ($1, $2, $3)`,
		userID, token, expiration,
	)
	return err
}
