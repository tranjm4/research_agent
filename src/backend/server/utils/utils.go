package utils

import (
	"fmt"
	"os"

	"github.com/joho/godotenv"
)

func GetServerUrl(path string) string {
	godotenv.Load(path)
	serverUrl := os.Getenv("SERVER_URL")
	serverPort := os.Getenv("SERVER_PORT")
	if serverUrl == "" {
		serverUrl = "http://localhost:8080" // Default value if not set
	}
	if serverPort == "" {
		serverPort = "8080" // Default value if not set
	}
	return fmt.Sprintf("%s:%s", serverUrl, serverPort)
}

func GetDBUrl(path string) string {
	godotenv.Load(path)
	user := os.Getenv("POSTGRES_USER")
	password := os.Getenv("POSTGRES_PASSWORD")
	dbname := os.Getenv("POSTGRES_DB")
	port := os.Getenv("DB_PORT")
	return fmt.Sprintf("postgres://%s:%s@localhost:%s/%s", user, password, port, dbname)
}
