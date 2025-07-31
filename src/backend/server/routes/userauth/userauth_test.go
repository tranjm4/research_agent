package userauth

import (
	"testing"

	"github.com/stretchr/testify/assert"

	"database/sql"
	"net/http"

	"bytes"
	"encoding/json"
	"server/utils"

	_ "github.com/jackc/pgx/v5/stdlib"
)

func TestDBConnection(t *testing.T) {
	assert := assert.New(t)
	// Here you would typically mock the database connection or use a test database
	assert.NotNil(assert, "Database connection should not be nil")

	dbUrl := utils.GetDBUrl("../../.env")
	assert.NotEmpty(dbUrl, "Database URL should not be empty")

	// Attempt to connect to the database
	db, err := sql.Open("pgx", dbUrl)
	assert.NoError(err, "Should connect to the database without error")
	defer db.Close()

	err = db.Ping()
	assert.NoError(err, "Should ping the database without error")
	t.Log("[PASSED] Database connection successful")
}

// Test for postLogin function's failure on empty input
// Expects the function to return an http.StatusBadRequest error (400)
func TestPostLoginFailsOnEmptyInput(t *testing.T) {
	assert := assert.New(t)
	serverUrl := utils.GetServerUrl("../../.env")
	assert.NotEmpty(serverUrl, "Server URL should not be empty")

	client := &http.Client{}

	// Make mock HTTP request to postLogin
	request, err := prepareRequest("/auth/login", "", "", "")
	assert.NoError(err, "Should prepare request without error")
	assert.NotNil(request, "Request should not be nil")
	resp, err := client.Do(request)
	assert.NoError(err, "Should not return an error when making the request")
	defer resp.Body.Close()
	assert.Equal(http.StatusBadRequest, resp.StatusCode, "Expected HTTP status code 400 for empty input")

	request, err = prepareRequest("/auth/login", "testuser", "", "")
	assert.NoError(err, "Should prepare request without error")
	assert.NotNil(request, "Request should not be nil")
	resp, err = client.Do(request)
	assert.NoError(err, "Should not return an error when making the request")
	defer resp.Body.Close()
	assert.Equal(http.StatusBadRequest, resp.StatusCode, "Expected HTTP status code 400 for empty password")

	request, err = prepareRequest("/auth/login", "", "", "testpassword")
	assert.NoError(err, "Should prepare request without error")
	assert.NotNil(request, "Request should not be nil")
	resp, err = client.Do(request)
	assert.NoError(err, "Should not return an error when making the request")
	defer resp.Body.Close()
	assert.Equal(http.StatusBadRequest, resp.StatusCode, "Expected HTTP status code 400 for empty username")
}

func TestRegisterNewUser(t *testing.T) {
	assert := assert.New(t)
	serverUrl := utils.GetServerUrl("../../.env")
	assert.NotEmpty(serverUrl, "Server URL should not be empty")

	// Make mock HTTP request to postRegister
	// Create a new user with a unique username
	username := "testuser_1"
	email := "testuser_1@example.com"
	password := "testpassword_1"

	resp, err := registerUser(username, email, password)
	assert.NoError(err, "Should not return an error when making the request")
	assert.NotNil(resp, "Response should not be nil")
	assert.NotEmpty(resp.Body, "Response body should not be empty")
	assert.NotEqual(resp.StatusCode, http.StatusInternalServerError, "Expected no internal server error")
	defer resp.Body.Close()

	assert.Equal(http.StatusCreated, resp.StatusCode, "Expected HTTP status code 201 for successful registration")

	// Verify the user was created in the database
	dbUrl := utils.GetDBUrl("../../.env")
	db, err := sql.Open("pgx", dbUrl)
	assert.NoError(err, "Should connect to the database without error")
	defer db.Close()

	var userExists bool
	err = db.QueryRow(`SELECT COUNT(*) > 0 FROM users WHERE username = $1`, username).Scan(&userExists)
	assert.NoError(err, "Should query the database without error")
	assert.True(userExists, "User should exist in the database after registration")
	t.Log("User registration successful")

	// Clean up: delete the test user
	deleteQuery := `DELETE FROM users WHERE username = $1`
	_, err = db.Exec(deleteQuery, username)
	assert.NoError(err, "Should delete the test user without error")
	t.Log("Test user deleted successfully")

	// Verify the user was deleted
	err = db.QueryRow(`SELECT COUNT(*) > 0 FROM users WHERE username = $1`, username).Scan(&userExists)
	assert.NoError(err, "Should query the database without error")
	assert.False(userExists, "User should not exist in the database after deletion")

}

func TestDuplicateRegisterFails(t *testing.T) {
	assert := assert.New(t)
	serverUrl := utils.GetServerUrl("../../.env")
	assert.NotEmpty(serverUrl, "Server URL should not be empty")

	username := "testuser_2"
	email := "testuser_2@example.com"
	password := "testpassword_2"

	// Register a user
	resp, err := registerUser(username, email, password)
	assert.NoError(err, "Should not return an error when making the request")
	defer resp.Body.Close()

	assert.Equal(http.StatusCreated, resp.StatusCode, "Expected HTTP status code 201 for successful registration")

	// Attempt to register the same username again
	resp, err = registerUser(username, email, password)
	assert.NoError(err, "Should not return an error when making the request")
	defer resp.Body.Close()

	assert.Equal(http.StatusConflict, resp.StatusCode, "Expected HTTP status code 409 for duplicate registration")
	t.Log("Duplicate registration test passed")

	// Clean up: delete the test user
	dbUrl := utils.GetDBUrl("../../.env")
	db, err := sql.Open("pgx", dbUrl)
	assert.NoError(err, "Should connect to the database without error")
	defer db.Close()

	deleteQuery := `DELETE FROM users WHERE username = $1`
	_, err = db.Exec(deleteQuery, username)
	assert.NoError(err, "Should delete the test user without error")

}

func TestLoginWithValidCredentials(t *testing.T) {
	assert := assert.New(t)
	serverUrl := utils.GetServerUrl("../../.env")
	assert.NotEmpty(serverUrl, "Server URL should not be empty")

	// Register a user first
	username := "testuser_3"
	email := "testuser_3@example.com"
	password := "testpassword_3"

	resp, err := registerUser(username, email, password)
	assert.NoError(err, "Should prepare request without error")
	assert.NoError(err, "Should not return an error when making the request")
	defer resp.Body.Close()

	assert.Equal(http.StatusCreated, resp.StatusCode, "Expected HTTP status code 201 for successful registration")

	// Now attempt to login with the same credentials
	resp, err = loginUser(username, password)
	assert.NoError(err, "Should prepare request without error")
	assert.NoError(err, "Should not return an error when making the request")
	defer resp.Body.Close()

	assert.Equal(http.StatusOK, resp.StatusCode, "Expected HTTP status code 200 for successful login")
	t.Log("Login with valid credentials test passed")

	// Clean up: delete the test user
	dbUrl := utils.GetDBUrl("../../.env")
	db, err := sql.Open("pgx", dbUrl)
	assert.NoError(err, "Should connect to the database without error")
	defer db.Close()

	deleteQuery := `DELETE FROM users WHERE username = $1`
	_, err = db.Exec(deleteQuery, username)
	assert.NoError(err, "Should delete the test user without error")
}

// func TestLogoutRemovesUserSession(t *testing.T) {
// 	assert := assert.New(t)
// 	serverUrl := utils.GetServerUrl("../../.env")
// 	assert.NotEmpty(serverUrl, "Server URL should not be empty")

// 	// Register a user first
// 	username := "testuser_4"
// 	email := "testuser_4@example.com"
// 	password := "testpassword_4"

// 	resp, err := registerUser(username, email, password)
// 	assert.NoError(err, "Should prepare request without error")
// 	assert.Equal(http.StatusCreated, resp.StatusCode, "Expected HTTP status code 201 for successful registration")
// 	defer resp.Body.Close()

// 	// Now attempt to login with the same credentials
// 	resp, err = loginUser(username, password)
// 	assert.NoError(err, "Should prepare request without error")
// 	assert.Equal(http.StatusOK, resp.StatusCode, "Expected HTTP status code 200 for successful login")
// 	defer resp.Body.Close()

// 	// Now attempt to logout
// }

type RequestData struct {
	Username string `json:"Username"`
	Email    string `json:"Email"`
	Password string `json:"Password"`
}

func prepareRequest(path string, username string, email string, password string) (*http.Request, error) {
	data := map[string]string{
		"Username": username,
		"Email":    email,
		"Password": password,
	}
	jsonData, _ := json.Marshal(data)

	route := "http://localhost:8080" + path
	req, err := http.NewRequest("POST", route, bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")

	return req, nil
}

func registerUser(username string, email string, password string) (*http.Response, error) {
	client := &http.Client{}
	request, err := prepareRequest("/auth/register", username, email, password)
	if err != nil {
		return nil, err
	}
	return client.Do(request)
}

func loginUser(username string, password string) (*http.Response, error) {
	client := &http.Client{}
	request, err := prepareRequest("/auth/login", username, "", password)
	if err != nil {
		return nil, err
	}
	return client.Do(request)
}
