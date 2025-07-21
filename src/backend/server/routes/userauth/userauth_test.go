package userauth

import (
	"log"
	"testing"

	"github.com/stretchr/testify/assert"

	"database/sql"
	"net/http"
	"net/url"

	"server/utils"

	_ "github.com/jackc/pgx/v5/stdlib"
)

func TestDBConnection(t *testing.T) {
	assert := assert.New(t)
	// Here you would typically mock the database connection or use a test database
	assert.NotNil(assert, "Database connection should not be nil")

	dbUrl := utils.GetDBUrl("../../.env")
	log.Println("Database URL:", dbUrl)
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

	// Make mock HTTP request to postLogin
	form := setCredentials("", "", "")
	resp, err := http.PostForm(serverUrl+"/auth/login", form)
	assert.NoError(err, "Should not return an error when making the request")
	defer resp.Body.Close()
	assert.Equal(resp.StatusCode, http.StatusBadRequest, "Expected HTTP status code 400 for empty input")

	form = setCredentials("testuser", "", "")
	resp, err = http.PostForm(serverUrl+"/auth/login", form)
	assert.NoError(err, "Should not return an error when making the request")
	defer resp.Body.Close()
	assert.Equal(resp.StatusCode, http.StatusBadRequest, "Expected HTTP status code 400 for empty password")

	form = setCredentials("", "", "testpassword")
	resp, err = http.PostForm(serverUrl+"/auth/login", form)
	assert.NoError(err, "Should not return an error when making the request")
	defer resp.Body.Close()
	assert.Equal(resp.StatusCode, http.StatusBadRequest, "Expected HTTP status code 400 for empty input")

	t.Log("[PASSED] PostLogin fails on empty input test passed")
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
	form := setCredentials(username, email, password)

	resp, err := http.PostForm(serverUrl+"/auth/register", form)
	assert.NoError(err, "Should not return an error when making the request")
	defer resp.Body.Close()

	assert.Equal(resp.StatusCode, http.StatusCreated, "Expected HTTP status code 201 for successful registration")

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
	t.Log("[PASSED] User deletion successful")
}

func TestDuplicateRegisterFails(t *testing.T) {
	assert := assert.New(t)
	serverUrl := utils.GetServerUrl("../../.env")
	assert.NotEmpty(serverUrl, "Server URL should not be empty")

	// Register a user
	username := "testuser_2"
	email := "testuser_2@example.com"
	password := "testpassword_2"
	form := setCredentials(username, email, password)

	resp, err := http.PostForm(serverUrl+"/auth/register", form)
	assert.NoError(err, "Should not return an error when making the request")
	defer resp.Body.Close()

	assert.Equal(resp.StatusCode, http.StatusCreated, "Expected HTTP status code 201 for successful registration")

	// Attempt to register the same username again
	resp, err = http.PostForm(serverUrl+"/auth/register", form)
	assert.NoError(err, "Should not return an error when making the request")
	defer resp.Body.Close()

	assert.Equal(resp.StatusCode, http.StatusConflict, "Expected HTTP status code 409 for duplicate registration")
	t.Log("Duplicate registration test passed")

	// Clean up: delete the test user
	dbUrl := utils.GetDBUrl("../../.env")
	db, err := sql.Open("pgx", dbUrl)
	assert.NoError(err, "Should connect to the database without error")
	defer db.Close()

	deleteQuery := `DELETE FROM users WHERE username = $1`
	_, err = db.Exec(deleteQuery, username)
	assert.NoError(err, "Should delete the test user without error")
	t.Log("[PASSED] Test user deleted successfully")
}

func TestLoginWithValidCredentials(t *testing.T) {
	assert := assert.New(t)
	serverUrl := utils.GetServerUrl("../../.env")
	assert.NotEmpty(serverUrl, "Server URL should not be empty")

	// Register a user first
	username := "testuser_3"
	email := "testuser_3@example.com"
	password := "testpassword_3"
	form := setCredentials(username, email, password)

	resp, err := http.PostForm(serverUrl+"/auth/register", form)
	assert.NoError(err, "Should not return an error when making the request")
	defer resp.Body.Close()

	assert.Equal(resp.StatusCode, http.StatusCreated, "Expected HTTP status code 201 for successful registration")

	// Now attempt to login with the same credentials
	loginForm := setCredentials(username, email, password)

	resp, err = http.PostForm(serverUrl+"/auth/login", loginForm)
	assert.NoError(err, "Should not return an error when making the request")
	defer resp.Body.Close()

	assert.Equal(resp.StatusCode, http.StatusOK, "Expected HTTP status code 200 for successful login")
	t.Log("Login with valid credentials test passed")

	// Clean up: delete the test user
	dbUrl := utils.GetDBUrl("../../.env")
	db, err := sql.Open("pgx", dbUrl)
	assert.NoError(err, "Should connect to the database without error")
	defer db.Close()

	deleteQuery := `DELETE FROM users WHERE username = $1`
	_, err = db.Exec(deleteQuery, username)
	assert.NoError(err, "Should delete the test user without error")
	t.Log("[PASSED] Test user deleted successfully")
}

func setCredentials(username string, email string, password string) url.Values {
	form := url.Values{}
	form.Set("username", username)
	form.Set("email", email)
	form.Set("password", password)
	return form
}
