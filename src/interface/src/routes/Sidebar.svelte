<!-- 
interface/src/routes/Sidebar.svelte

This file defines the sidebar component of the application.
This includes the conversation list, user profile, navigation links,
and buttons for creating new conversations or logging out.
-->

<script lang="ts">
	import { onMount } from 'svelte';
	import { goto } from '$app/navigation';
	import { auth } from '../stores/auth.svelte';

	interface Conversation {
		id: string;
		title: string;
		createdAt: string;
		lastModified: string;
	}
	let conversations: Array<Conversation> = $state([]);

	let conversationTitle: string = $state('');
	let isLoading: boolean = $state(true);
	let showModal: boolean = $state(false);
	let isCreatingConversation: boolean = $state(false);

	let isFormValid: boolean = $derived(conversationTitle.trim() !== '' && showModal);

	const serverUrl = 'http://localhost:8080';

	onMount(() => {
		getConversationList();
	});

	$effect(() => {
		if (auth.isAuthenticated && auth.user) {
			getConversationList();
		} else {
			conversations = [];
		}
	});

	// $: if ($isAuthenticated) {
	// 	getConversationList();
	// } else {
	// 	conversations = [];
	// }

	function getConversationList() {
		// Get the user's conversation history from the server

		// The server should recognize the user's session token and
		// return the conversation history for that user.
		// If the user is not logged in, we should receive a 401 Unauthorized response.
		// And we should redirect the user to the login page.
		let response: Promise<Response> = fetch(`${serverUrl}/chat/conversations`, {
			method: 'GET',
			headers: {
				'Content-Type': 'application/json'
			},
			credentials: 'include', // Include cookies for session management
			mode: 'cors' // Ensure CORS is handled correctly
		});
		response
			.then((res) => {
				if (res.status === 401) {
					goto('/auth/login'); // Redirect to login if unauthorized
				} else if (!res.ok) {
					console.error('Error fetching conversations:', res.statusText);
					alert('Failed to load conversations. Please try again later.');
					return;
				}
				return res.json();
			})
			.then((data) => {
				console.log(data);
				conversations = data.conversations || [];
				// TODO: error getting json data
				if (!Array.isArray(conversations)) {
					console.error('Invalid conversations data format:', conversations);
					alert('Failed to load conversations. Please try again later.');
					return;
				}
			})
			.catch((error) => {
				console.error('Error during fetch:', error);
				alert('An error occurred while fetching conversations. Please try again later.');
			});
	}

	function openModal() {
		showModal = true;
		conversationTitle = '';
	}

	function closeModal() {
		showModal = false;
		conversationTitle = '';
		isCreatingConversation = false;
	}

	async function createConversation() {
		if (!conversationTitle.trim()) {
			alert('Please enter a title for the conversation.');
			return;
		}
		isCreatingConversation = true;
		try {
			const response = await fetch(`${serverUrl}/chat/create`, {
				method: 'POST',
				headers: {
					'Content-Type': 'application/json'
				},
				credentials: 'include',
				body: JSON.stringify({ title: conversationTitle.trim() })
			});

			if (!response.ok) {
				if (response.status === 401) {
					goto('/auth/login'); // Redirect to login if unauthorized
					return;
				} else {
					throw new Error('Failed to create conversation');
				}
			}

			const data = await response.json();
			console.log(data);
			const newConversation: Conversation = {
				id: data.id,
				title: conversationTitle.trim(),
				createdAt: data.createdAt,
				lastModified: data.lastModified
			};
			conversations = [...conversations, newConversation];

			closeModal();
			goto(`/chat/${newConversation.id}`);
		} catch (error) {
			console.error('Error creating conversation:', error);
			alert('Failed to create conversation. Please try again.');
		} finally {
			isCreatingConversation = false;
		}
	}

	function handleKeydown(event: KeyboardEvent) {
		if (event.key === 'Escape') {
			closeModal();
		} else if (event.key === 'Enter' && isFormValid) {
			event.preventDefault();
			createConversation();
		}
	}

	function handleModalClick(event: MouseEvent) {
		if (event.target === event.currentTarget) {
			closeModal();
		}
	}
</script>

<svelte:window on:keydown={handleKeydown} />

<div class="sidebar">
	<h2>Conversations</h2>
	<ul>
		{#each conversations as conversation}
			<li>
				<a href={`/chat/${conversation.id}`}>{conversation.title}</a>
			</li>
		{/each}
	</ul>
	<button onclick={openModal}>New Conversation</button>
	<button onclick={() => goto('/logout')}>Logout</button>
</div>

{#if showModal}
	<!-- svelte-ignore a11y_no_static_element_interactions -->
	<!-- svelte-ignore a11y_click_events_have_key_events -->
	<div class="modal-backdrop" onclick={handleModalClick}>
		<div class="modal">
			<div class="modal-header">
				<h3>Create New Conversation</h3>
				<button class="close-btn" onclick={closeModal}>&times;</button>
			</div>
			<div class="modal-body">
				<form onsubmit={createConversation}>
					<label for="conversation-title">Title:</label>
					<input
						type="text"
						id="conversation-title"
						bind:value={conversationTitle}
						placeholder="Enter conversation title..."
						disabled={isCreatingConversation}
					/>
					<div class="modal-actions">
						<button type="button" onclick={closeModal} disabled={isCreatingConversation}
							>Cancel</button
						>
						<button type="submit" disabled={isCreatingConversation}>
							{isCreatingConversation ? 'Creating...' : 'Create'}
						</button>
					</div>
				</form>
			</div>
		</div>
	</div>
{/if}

<style>
	.sidebar {
		width: 250px;
		background-color: #f4f4f4;
		padding: 1rem;
		box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
	}

	h2 {
		margin-top: 0;
	}

	ul {
		list-style-type: none;
		padding: 0;
	}

	li {
		margin-bottom: 0.5rem;
	}

	a {
		text-decoration: none;
		color: #333;
	}

	a:hover {
		text-decoration: underline;
	}

	button {
		margin-top: 1rem;
		width: 100%;
		padding: 0.5rem;
		background-color: #007bff;
		color: white;
		border: none;
		border-radius: 4px;
		cursor: pointer;
	}

	button:hover {
		background-color: #0056b3;
	}
	.modal-backdrop {
		position: fixed;
		top: 0;
		left: 0;
		width: 100%;
		height: 100%;
		background-color: rgba(0, 0, 0, 0.5);
		display: flex;
		align-items: center;
		justify-content: center;
	}
	.modal {
		background-color: white;
		padding: 1rem;
		border-radius: 8px;
		box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
		width: 90%;
		max-width: 500px;
	}
	.modal-header {
		display: flex;
		justify-content: space-between;
		align-items: center;
		margin-bottom: 1rem;
	}
	/* .close-btn {
		background: none;
		border: none;
		font-size: 1.5rem;
		cursor: pointer;
	} */
	.modal-body {
		display: flex;
		flex-direction: column;
	}
	.modal-body label {
		margin-bottom: 0.5rem;
	}
	.modal-body input {
		padding: 0.5rem;
		border: 1px solid #ccc;
		border-radius: 4px;
		margin-bottom: 1rem;
	}
	.modal-actions {
		display: flex;
		justify-content: space-between;
	}
	.modal-actions button {
		padding: 0.5rem 1rem;
		background-color: #007bff;
		color: white;
		border: none;
		border-radius: 4px;
		cursor: pointer;
	}
	.modal-actions button:hover {
		background-color: #0056b3;
	}
	.modal-actions button:disabled {
		background-color: #ccc;
		cursor: not-allowed;
	}
	.modal-actions button:disabled:hover {
		background-color: #ccc;
	}
</style>
