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
	import { postLogin } from '../auth';
	import { auth } from '../../../stores/auth.svelte';

	let usernameOrEmail: string = '';
	let password: string = '';

	$: isFormValid = usernameOrEmail.trim() !== '' && password.trim() !== '';

	function handleSubmit(event: Event) {
		event.preventDefault();

		if (usernameOrEmail.trim() === '' || password.trim() === '') {
			alert('Please fill in both fields.');
			return;
		}

		let response: Promise<Response> = postLogin(usernameOrEmail, password);
		response
			.then((res) => {
				if (!res.ok) {
					console.error('Error logging in:', res.statusText);
					alert('Login failed. Please check your credentials and try again.');
					return;
				} else {
					console.log('Login successful');
					// Load the conversation list
					auth.setUser({
						name: usernameOrEmail
					});
					goto('/chat'); // Redirect to chat page after successful login
				}
			})
			.catch((error) => {
				console.error('Error during fetch:', error);
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
	<button type="submit" disabled={!isFormValid}>Log In</button>
</form>
