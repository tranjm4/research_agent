<script lang="ts">
	import { onMount, tick } from 'svelte';

	let postUrl: string = 'http://localhost:8080/auth/register';

	import { goto } from '$app/navigation';

	let username: string = '';
	let email: string = '';
	let password: string = '';
	let confirmPassword: string = '';

	$: isPasswordsMatch = password === confirmPassword;
	$: isFormValid =
		username.trim() !== '' && email.trim() !== '' && password.trim() !== '' && isPasswordsMatch;
	$: passwordStrength = checkPasswordStrength();

	interface UserData {
		username: string;
		email: string;
		password: string;
	}

	function checkPasswordStrength(): string {
		if (password.length === 0) {
			return '';
		} else if (password.length < 8) {
			return 'Weak';
		} else if (password.length < 12) {
			return 'Moderate';
		} else {
			return 'Strong';
		}
	}

	// handleSubmit
	function handleSubmit(event: Event) {
		event.preventDefault();
		// Handle form submission logic here
		if (isFormValid) {
			const registerData: UserData = {
				username: username,
				email: email,
				password: password
			};

			// send the data to the server
			let response: Promise<Response> = fetch(postUrl, {
				method: 'POST',
				headers: {
					'Content-Type': 'application/json'
				},
				credentials: 'include', // Include cookies in the request
				mode: 'cors', // Ensure CORS is enabled
				body: JSON.stringify(registerData)
			});

			response.then((res) => {
				if (!res.ok) {
					console.error('Error signing up:', res.statusText);
					alert('Sign up failed. Please check your details and try again.');
					return;
				} else {
					console.log('Sign up successful');
					goto('/chat');
				}
			}); // then send the user to the chat page
			response.catch((error) => {
				console.error('Error during fetch:', error);
			});
		} else {
			console.error('Form is invalid');
		}
	}

	onMount(() => {
		// Any initialization logic can go here
	});
</script>

<div class="signup-container">
	<h1>Sign Up</h1>
	<form on:submit={handleSubmit}>
		<label>
			Username:
			<input type="text" bind:value={username} required />
		</label>
		<label>
			Email:
			<input type="email" bind:value={email} required />
		</label>
		<label>
			Password:
			<input type="password" bind:value={password} required />
			{#if password !== ''}
				<span>Password Strength: {passwordStrength}</span>
			{/if}
		</label>
		<span class="pass-strength">{passwordStrength}</span>
		{#if passwordStrength === 'Weak'}
			<span class="error">Password must be at least 8 characters long</span>
		{/if}
		Confirm Password:
		<input type="password" bind:value={confirmPassword} required />
		{#if !isPasswordsMatch}
			<span class="error">Passwords do not match</span>
		{/if}
		{#if confirmPassword !== '' && password !== '' && !isPasswordsMatch}
			<span class="error">Passwords do not match</span>
		{/if}

		{#if isFormValid}
			<span class="success">Form is valid</span>
		{:else}
			<span class="error">Please fill out all fields correctly</span>
		{/if}

		<button type="submit" disabled={!isFormValid}>Sign Up</button>
	</form>
</div>
