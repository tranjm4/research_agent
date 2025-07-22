<!-- 
src/routes/auth/login/+page.svelte

This page is responsible for the login functionality of the application.
It handles user input for username and password, validates the input,
and sends a login request to the server. If the login is successful,
it redirects the user to the chat page. If there are errors, it displays
them to the user.
-->

<script lang="ts">
	import { goto } from '$app/navigation';

	let usernameOrEmail: string = '';
	let password: string = '';

	interface LoginData {
		username: string;
		password: string;
	}

	let postUrl: string = 'http://localhost:8080/auth/login';

	function handleSubmit(event: Event) {
		event.preventDefault();

		if (usernameOrEmail.trim() === '' || password.trim() === '') {
			alert('Please fill in both fields.');
			return;
		}
		const loginData: LoginData = {
			username: usernameOrEmail,
			password: password
		};

		let response: Promise<Response> = fetch(postUrl, {
			method: 'POST',
			headers: {
				'Content-Type': 'application/json'
			},
			body: JSON.stringify(loginData),
			mode: 'cors', // Ensure CORS is enabled
			credentials: 'include' // Include cookies in the request
		});

		response.then((res) => {
			if (!res.ok) {
				console.error('Login failed:', res.statusText);
				alert('Login failed. Please check your credentials and try again.');
			} else {
				console.log('Registration successful');
				goto('/chat');
			}
		});
		response.catch((error) => {
			console.error('Error during login:', error);
			alert('An error occurred while trying to log in. Please try again later.');
		});
	}
</script>

<form on:submit|preventDefault={handleSubmit}>
	<label for="username">Username or Email:</label>
	<input
		id="username"
		type="text"
		placeholder="Enter your username or email"
		bind:value={usernameOrEmail}
		required
	/>
	<label for="password">Password:</label>
	<input
		id="password"
		type="password"
		placeholder="Enter your password"
		bind:value={password}
		required
	/>
	<button type="submit">Log In</button>
</form>
