<script lang="ts">
	import { onMount, tick } from 'svelte';
	import { postRegister } from '../auth';

	import { goto } from '$app/navigation';
	import { auth } from '../../../stores/auth.svelte';

	let username: string = $state('');
	let email: string = $state('');
	let password: string = $state('');
	let confirmPassword: string = $state('');

	let isPasswordsMatch: boolean = $derived(password === confirmPassword);
	let isFormValid: boolean = $derived(
		username.trim() !== '' && email.trim() !== '' && password.trim() !== '' && isPasswordsMatch
	);
	let passwordStrength: string = $derived(checkPasswordStrength());

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

	interface RegisterData {
		username: string;
		email: string;
		password: string;
	}
	const postUrl = 'http://localhost:8080/auth/register';

	// Make a POST request to the registration endpoint
	// and handle the response.
	function handleSubmit(event: Event) {
		event.preventDefault();
		// Handle form submission logic here
		if (isFormValid) {
			const postData: RegisterData = {
				username,
				email,
				password
			};
			let response: Promise<Response> = fetch(postUrl, {
				method: 'POST',
				headers: {
					'Content-Type': 'application/json'
				},
				body: JSON.stringify(postData),
				credentials: 'include',
				mode: 'cors'
			});
			// Handle the response
			response.then((res) => {
				if (!res.ok) {
					console.error('Error signing up:', res.statusText);
					alert('Sign up failed. Please check your details and try again.');
					return;
				} else {
					console.log('Sign up successful');
					// Update the auth store to reflect the new user
					auth.setUser({
						name: username
					});
					goto('/chat'); // Redirect to chat page after successful sign up
				}
			});
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
	<form onsubmit={handleSubmit}>
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
