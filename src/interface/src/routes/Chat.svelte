<script lang="ts">
	import { onMount, tick } from 'svelte';
	import { marked } from 'marked';

	const API_BASE_URL: string = 'http://localhost:8080';

	interface Message {
		id: number;
		content: string;
		sender: 'user' | 'assistant' | 'error';
		timestamp: Date;
		isStreaming?: boolean;
	}

	interface APIRequest {
		content: string;
		conversationId: string;
	}

	interface APIResponse {
		type: 'token' | 'end';
		content: string;
	}

	let props = $props();

	let conversationId: string = $derived(props.conversationId || '');
	let propMessages: Message[] = $derived(props.messages || []); // receive the list of messages from the parent component

	let messageCounter: number = $state(0); // counter to track the number of messages and for identification

	// reactive variables
	let currentMessage: string = $state('');
	let isLoading: boolean = $state(false);
	let messages: Message[] = $state([]);
	let streamingContent: string = $state(''); // to accumulate streaming content
	let streamingMessageId: number | null = $state(null);
	let isStreaming: boolean = $state(false); // to track if a message is currently streaming

	let chatContainer: HTMLDivElement | null = $state(null);
	let messageInput: HTMLInputElement | null = $state(null); // to bind the input element

	let canSend: boolean = $derived(currentMessage.trim() !== '' && !isLoading);
	let buttonText: string = $derived(
		isLoading ? (isStreaming ? 'Streaming...' : 'Sending...') : 'Send'
	);
	let inputPlaceholder: string = $derived(
		isLoading ? 'Please wait...' : 'Type your message here...'
	);

	onMount(() => {
		scrollToBottom();
	});

	$effect(() => {
		if (conversationId) {
			console.log('Conversation ID changed:', conversationId);

			// Clear the messages
			messages = [];
			isStreaming = false;
			streamingContent = '';
			streamingMessageId = null;

			// Load the messages for the conversation
			// This should be provided by the parent component
			messages = propMessages.map((msg) => ({
				...msg,
				isStreaming: false // Ensure all messages are not streaming initially
			}));
		}
	});

	function addMessage(content: string, sender: Message['sender'] = 'user'): Message {
		const message: Message = {
			id: messageCounter++,
			content,
			sender,
			timestamp: new Date(),
			isStreaming: sender === 'assistant' && content === '' // If sender is "assistant" and content is empty, it"s a streaming message
		};

		messages = [...messages, message];
		return message;
	}

	function updateStreamingMessage(
		messageId: number,
		newContent: string,
		isComplete: boolean = false
	): void {
		messages = messages.map((msg) => {
			if (msg.id === messageId) {
				// parse the markdown content if needed
				const markedContent: Promise<string> | string = marked.parse(newContent);
				// purify the content to prevent XSS attacks
				msg;
				return {
					...msg,
					content: newContent,
					isStreaming: !isComplete
				};
			}
			return msg;
		});
	}

	function completeStreaming(messageId: number): void {
		messages = messages.map((msg) => {
			if (msg.id === messageId) {
				return { ...msg, isStreaming: false };
			}
			return msg;
		});
		isStreaming = false;
		streamingContent = '';
		streamingMessageId = null;
	}

	async function sendMessageStream(): Promise<void> {
		if (!canSend) return; // If cannot send, do nothing

		// Add user message to chat
		const userMessage: string = currentMessage.trim();
		addMessage(userMessage, 'user');
		currentMessage = ''; // reset input field to empty string
		streamingContent = ''; // reset streaming content

		isLoading = true; // set loading state to true until response is received
		await tick(); // Ensure the DOM updates before making the API call
		scrollToBottom();

		const botMessage: Message = addMessage('', 'assistant'); // Add a bot message with empty content
		streamingMessageId = botMessage.id; // Store the ID of the streaming message

		try {
			console.log('Sending message:', userMessage, 'to conversation ID:', conversationId);
			const requestBody: APIRequest = {
				content: userMessage,
				conversationId: conversationId
			};

			const response: Response = await fetch(`${API_BASE_URL}/chat/send`, {
				method: 'POST',
				headers: {
					'Content-Type': 'application/json'
				},
				body: JSON.stringify(requestBody),
				credentials: 'include',
				mode: 'cors'
			});

			if (!response.ok) {
				throw new Error(`HTTP Error; status: ${response.status}`);
			}
			const reader = response.body?.getReader();
			if (!reader) {
				throw new Error('Response body is not readable');
			}

			const decoder = new TextDecoder();
			let accumulatedContent = '';

			try {
				while (true) {
					const { done, value } = await reader.read();
					if (done) {
						break;
					}

					const chunk: string = decoder.decode(value, { stream: true });
					const lines = chunk.split('\n');
					lines.forEach((token) => {
						if (token.trim()) {
							try {
								const tokenJSON: APIResponse = JSON.parse(token);
								const tokenText: string = tokenJSON.content;
								if (tokenText === '') {
									// accumulatedContent += ""; // Handle empty content
									// streamingContent += "<br/>";
								} else {
									accumulatedContent += tokenText;
									streamingContent += tokenText;
								}
							} catch (error) {
								console.error('Error parsing JSON line:', error, 'token:', token);
							}
						}
					});
					updateStreamingMessage(streamingMessageId, streamingContent, false);
				}
				completeStreaming(streamingMessageId);
			} finally {
				reader.releaseLock();
			}
		} catch (error) {
			console.error('Error sending message:', error);
			if (streamingMessageId) {
				updateStreamingMessage(streamingMessageId, 'Sorry, something went wrong.', true);
				completeStreaming(streamingMessageId);
			} else {
				addMessage('Sorry, something went wrong.', 'error');
			}
		} finally {
			isLoading = false;
			streamingMessageId = null;
			await tick();
			scrollToBottom();
		}
	}

	// async function sendMessage(event: Event): Promise<void> {
	// 	event.preventDefault();
	// 	if (!canSend) return; // If cannot send, do nothing

	// 	if (isStreaming) {
	// 		// If currently streaming, stop the streaming and send the accumulated content
	// 		if (streamingMessageId !== null) {
	// 			await completeStreaming(streamingMessageId);
	// 		}
	// 	} else {
	// 		await sendMessageStream();
	// 	}
	// }

	function focusInput(): void {
		messageInput?.focus();
	}

	function scrollToBottom(): void {
		if (chatContainer) {
			chatContainer.scrollTop = chatContainer.scrollHeight;
		}
	}
</script>

<div class="chat-container">
	<div class="messages" bind:this={chatContainer}>
		{#each messages as message}
			<div class="message {message.sender}" class:streaming={message.isStreaming}>
				<span class="content">
					{message.content}
					{#if message.isStreaming}
						<span class="cursor">▊</span>
					{/if}
				</span>
				<span class="timestamp">
					{new Date(message.timestamp).toLocaleTimeString()}
				</span>
			</div>
		{/each}

		{#if isLoading}
			<!-- <div class="message bot">
				<span class="content">Connecting...</span>
			</div> -->
		{/if}
	</div>

	<form onsubmit={sendMessageStream} class="input-form">
		<label for="search">Chat:</label>
		<input
			id="search"
			type="text"
			bind:this={messageInput}
			bind:value={currentMessage}
			placeholder={inputPlaceholder}
			onkeydown={(e) => {
				if (e.key === 'Enter' && !e.shiftKey) {
					e.preventDefault();
					sendMessageStream();
				}
			}}
			disabled={isLoading}
		/>
		<button
			type="submit"
			disabled={!canSend}
			class:loading={isLoading}
			class:streaming={isStreaming}
		>
			{buttonText}
		</button>
	</form>

	{#if isStreaming}
		<div class="streaming-info">
			<small>
				Streaming... ({streamingContent.length} characters)

				{#if streamingContent}
					<button onclick={() => focusInput()}> Continue typing </button>
				{/if}
			</small>
		</div>
	{/if}
</div>

<style>
	.message.streaming .content {
		border-left: 3px solid #007bff;
		padding-left: 0.5rem;
	}

	.cursor {
		animation: blink 1s infinite;
		color: #007bff;
		font-weight: bold;
	}

	@keyframes blink {
		0%,
		50% {
			opacity: 1;
		}
		51%,
		100% {
			opacity: 0;
		}
	}

	.button.loading {
		background-color: #6c757d;
	}

	.button.streaming {
		background-color: #28a745;
		animation: pulse 2s infinite;
	}

	@keyframes pulse {
		0% {
			opacity: 1;
		}
		50% {
			opacity: 0.7;
		}
		100% {
			opacity: 1;
		}
	}

	.streaming-info {
		padding: 0.5rem 1rem;
		background-color: #e7f3ff;
		border-top: 1px solid #bee5eb;
		font-size: 0.875rem;
		color: #0c5460;
	}

	/* Base styles */
	.chat-container {
		display: flex;
		flex-direction: column;
		height: 100vh;
		max-width: 800px;
		margin: 0 auto;
	}

	.messages {
		flex: 1;
		overflow-y: auto;
		padding: 1rem;
	}

	.message {
		margin-bottom: 1rem;
		padding: 0.75rem;
		border-radius: 8px;
	}

	.message.user {
		background-color: #007bff;
		color: white;
		margin-left: auto;
		max-width: 70%;
	}

	.message.assistant {
		background-color: #f8f9fa;
		border: 1px solid #dee2e6;
		max-width: 70%;
	}

	.input-form {
		display: flex;
		gap: 0.5rem;
		padding: 1rem;
		border-top: 1px solid #dee2e6;
	}

	.input-form input {
		flex: 1;
		padding: 0.75rem;
		border: 1px solid #ced4da;
		border-radius: 4px;
	}

	.input-form button {
		padding: 0.75rem 1.5rem;
		background-color: #007bff;
		color: white;
		border: none;
		border-radius: 4px;
		cursor: pointer;
	}

	.input-form button:disabled {
		opacity: 0.6;
		cursor: not-allowed;
	}
</style>
