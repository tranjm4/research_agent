package main

import (
	"context"
	"net/http"

	"fmt"
	"log"
	"os"
	"time"

	"server/routes/chat"
	"server/routes/userauth"
	"server/utils"

	"database/sql"

	"github.com/go-chi/chi/v5"
	"github.com/go-chi/chi/v5/middleware"
	"github.com/go-chi/cors"
	_ "github.com/jackc/pgx/v5/stdlib" // Import the PostgreSQL driver
)

func main() {
	r := chi.NewRouter()
	serverURL := os.Getenv("CLIENT_URL")
	r.Use(middleware.RequestID)
	r.Use(middleware.RealIP)
	r.Use(middleware.Logger)
	r.Use(middleware.Recoverer)
	r.Use(cors.Handler(cors.Options{
		AllowedOrigins:   []string{serverURL},
		AllowedMethods:   []string{"GET", "POST", "PUT", "OPTIONS"},
		AllowedHeaders:   []string{"Accept", "Authorization", "Content-Type"},
		AllowCredentials: true,
	}))

	// Set timeout for requests after 10 minutes
	r.Use(middleware.Timeout(10 * time.Minute))

	// Initialize the database connection
	// Connect to the PostgreSQL database
	dsn := utils.GetDBUrl(".env")
	db, err := sql.Open("pgx", dsn)
	if err != nil {
		log.Fatalf("Failed to connect to the database: %v", err)
	}

	defer db.Close() // Cleanup function when main exits

	// Ping the database to ensure connection is established
	ctx := context.Background()
	if err := db.PingContext(ctx); err != nil {
		log.Fatalf("failed to ping DB %v", err)
	}

	// Register routes
	userauth.RegisterRoutes(r, db)
	chat.RegisterRoutes(r, db)
	// ai.RegisterRoutes(r, db)

	fmt.Println("Connected to PostgreSQL database")
	fmt.Println("Server is running on", utils.GetServerUrl(".env"))
	serverPort := os.Getenv("SERVER_PORT")
	if serverPort == "" {
		log.Printf("Failed to retrieve server port from environment variables; using default (8080)")
	}
	http.ListenAndServe(":"+serverPort, r)
}
