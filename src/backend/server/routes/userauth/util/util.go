package util

import (
	"database/sql"
	"net/http"
	"os"
	"time"

	"github.com/google/uuid"

	"github.com/golang-jwt/jwt/v5"
	"github.com/joho/godotenv"
)

// generateSessionToken generates a session token for the user
// This function should create a secure token, store it in the database,
// and return it to the client.
func GenerateSessionToken(username string, db *sql.DB, secretKey []byte) (string, error) {
	// Create a new token
	token := jwt.NewWithClaims(jwt.SigningMethodHS256, jwt.MapClaims{
		"Subject":   username,
		"ExpiresAt": jwt.NewNumericDate(time.Now().Add(24 * time.Hour)), // Set expiration to 24 hours from now
		"IssuedAt":  jwt.NewNumericDate(time.Now()),
		"jti":       uuid.NewString(), // Generate a unique identifier for the token
	})

	signedToken, err := token.SignedString(secretKey)
	if err != nil {
		return "", err
	}

	// Insert the session token into the database
	err = InsertSessionToDB(db, username, signedToken)
	if err != nil {
		return "", err
	}

	// Return the token
	return signedToken, nil
}

func InsertSessionToDB(db *sql.DB, username string, token string) error {
	// get the user ID from the users table
	tx, err := db.Begin()
	if err != nil {
		return err
	}
	defer tx.Rollback()
	var userID int
	err = tx.QueryRow("SELECT id FROM users WHERE username = $1", username).Scan(&userID)
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
	if err != nil {
		return err
	}
	return tx.Commit()
}

func RemoveSessionFromDB(db *sql.DB, username string) error {
	// get the user ID from the users table
	tx, err := db.Begin()
	if err != nil {
		return err
	}
	defer tx.Rollback() // Ensure the transaction is rolled back in case of an error

	var userID int
	err = tx.QueryRow("SELECT id FROM users WHERE username = $1", username).Scan(&userID)
	if err != nil {
		return err
	}

	// Remove the session token from the session_tokens table
	_, err = tx.Exec("DELETE FROM session_tokens WHERE user_id = $1", userID)

	if err != nil {
		return err
	}
	return tx.Commit()
}

func ReplaceSessionToken(db *sql.DB, username string, newToken string) error {
	tx, err := db.Begin()
	if err != nil {
		return err
	}
	defer tx.Rollback()

	var userID int
	err = tx.QueryRow(`SELECT id FROM users WHERE username = $1`, username).Scan(&userID)
	if err != nil {
		return err
	}
	// Remove the existing session token for the user
	_, err = tx.Exec(`DELETE FROM session_tokens WHERE user_id = $1`, userID)
	if err != nil {
		return err
	}

	// Insert the new session token into the session_tokens table
	expiration := time.Now().Add(24 * time.Hour) // Set expiration to 24 hours from now
	_, err = tx.Exec(`INSERT INTO session_tokens (user_id, token, expires_at) 
		VALUES ($1, $2, $3)`,
		userID, newToken, expiration,
	)
	if err != nil {
		return err
	}
	return tx.Commit()
}

func ExtractTokenFromRequest(r *http.Request) string {
	cookie, err := r.Cookie("session_token")
	if err != nil {
		return ""
	}
	return cookie.Value
}

func VerifyJWT(tokenString string) (*jwt.RegisteredClaims, error) {
	if tokenString == "" {
		return nil, jwt.ErrTokenMalformed
	}
	secretKey, err := LoadSecretKey("../../.env")
	if err != nil {
		return nil, err
	}

	token, err := jwt.ParseWithClaims(tokenString, &jwt.RegisteredClaims{}, func(token *jwt.Token) (interface{}, error) {
		return secretKey, nil
	})
	if err != nil || !token.Valid {
		return nil, jwt.ErrTokenMalformed
	}

	claims, ok := token.Claims.(*jwt.RegisteredClaims)
	if !ok {
		return nil, jwt.ErrTokenMalformed
	}

	return claims, nil
}

func LoadSecretKey(path string) ([]byte, error) {
	godotenv.Load(path) // Adjust the path as necessary
	secretKey := os.Getenv("JWT_SECRET")
	if secretKey == "" {
		return nil, http.ErrNoCookie
	}
	return []byte(secretKey), nil
}

func VerifyUserSession(db *sql.DB, r *http.Request) (bool, error) {
	tokenString := ExtractTokenFromRequest(r)
	if tokenString == "" {
		return false, http.ErrNoCookie
	}

	// Check if the session token exists and is valid
	var exists bool
	err := db.QueryRow("SELECT EXISTS(SELECT 1 FROM session_tokens WHERE token = $1 AND is_valid = true)", tokenString).Scan(&exists)
	if err != nil {
		return false, err
	}
	return exists, nil
}
