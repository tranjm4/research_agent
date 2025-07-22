<!-- src/routes/auth/+layout.svelte -->
<!-- This file is part of the authentication layout for the application. -->
<script lang="ts">
	import Header from '../Header.svelte';
	let { children } = $props();

	function handleSubmit(event: Event) {
		event.preventDefault();
		// Handle form submission logic here
	}
</script>

<div class="auth-layout">
	<main>
		{@render children()}
	</main>
</div>

<style>
	/* .auth-layout {
		display: flex;
		flex-direction: column;
		min-height: 100vh;
	} */
</style>
