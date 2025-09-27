package utils

import (
	"fmt"
	"log"
	"os"

	"github.com/joho/godotenv"
	"github.com/sqids/sqids-go"
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
	return fmt.Sprintf("postgres://%s:%s@ra-psql:%s/%s", user, password, port, dbname)
}

type SqidsManager struct {
	userEncoder         *sqids.Sqids
	messageEncoder      *sqids.Sqids
	conversationEncoder *sqids.Sqids
	sessionTokenEncoder *sqids.Sqids
}

func NewSqidsManager() (*SqidsManager, error) {
	alphabet := "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"

	userEncoder, err := buildEncoder(alphabet, 8, []string{"admin", "root", "superuser"})
	if err != nil {
		panic(fmt.Sprintf("Failed to create user encoder: %v", err))
	}
	messageEncoder, err := buildEncoder(alphabet, 16, []string{"message", "msg", "text"})
	if err != nil {
		panic(fmt.Sprintf("Failed to create message encoder: %v", err))
	}
	conversationEncoder, err := buildEncoder(alphabet, 12, []string{"conversation", "chat", "thread"})
	if err != nil {
		panic(fmt.Sprintf("Failed to create conversation encoder: %v", err))
	}
	sessionTokenEncoder, err := buildEncoder(alphabet, 32, []string{"session", "token", "auth", "userauth", "login", "register", "logout"})
	if err != nil {
		panic(fmt.Sprintf("Failed to create session token encoder: %v", err))
	}

	return &SqidsManager{
		userEncoder:         userEncoder,
		messageEncoder:      messageEncoder,
		conversationEncoder: conversationEncoder,
		sessionTokenEncoder: sessionTokenEncoder,
	}, nil
}

func buildEncoder(alphabet string, minLength uint8, blocklist []string) (*sqids.Sqids, error) {
	encoder, err := sqids.New(sqids.Options{
		Alphabet:  alphabet,
		MinLength: minLength,
		Blocklist: blocklist,
	})
	if err != nil {
		return nil, err
	}
	return encoder, nil
}

func encodeID(id int64, encoder *sqids.Sqids) (string, error) {
	encoded, err := encoder.Encode([]uint64{uint64(id)})
	if err != nil {
		log.Printf("Error encoding ID %d: %v", id, err)
		return "", err
	}
	return encoded, nil
}

func decodeID(sqid string, encoder *sqids.Sqids) (int64, error) {
	decoded := encoder.Decode(sqid)
	if len(decoded) == 0 {
		return 0, fmt.Errorf("Invalid sqid: %s", sqid)
	}
	return int64(decoded[0]), nil
}

func (sm *SqidsManager) EncodeUserID(id int64) (string, error) {
	return encodeID(id, sm.userEncoder)
}

func (sm *SqidsManager) EncodeMessageID(id int64) (string, error) {
	return encodeID(id, sm.messageEncoder)
}

func (sm *SqidsManager) EncoderConversationID(id int64) (string, error) {
	return encodeID(id, sm.conversationEncoder)
}

func (sm *SqidsManager) EncodeSessionTokenID(id int64) (string, error) {
	return encodeID(id, sm.sessionTokenEncoder)
}

func (sm *SqidsManager) DecodeUserID(sqid string) (int64, error) {
	return decodeID(sqid, sm.userEncoder)
}

func (sm *SqidsManager) DecodeMessageID(sqid string) (int64, error) {
	return decodeID(sqid, sm.messageEncoder)
}

func (sm *SqidsManager) DecodeConversationID(sqid string) (int64, error) {
	return decodeID(sqid, sm.conversationEncoder)
}

func (sm *SqidsManager) DecodeSessionTokenID(sqid string) (int64, error) {
	return decodeID(sqid, sm.sessionTokenEncoder)
}
