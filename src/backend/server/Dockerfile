# This Dockerfile is used to build the backend server image
FROM golang:1.24.5

WORKDIR /app

# Copy the Go modules
COPY go.mod go.sum ./
RUN go mod download

COPY . .

RUN go build -v -o ./server

CMD ["./server"]