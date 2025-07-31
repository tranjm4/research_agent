<!-- 
src/routes/chat/[slug]/+page.svelte 

This page is responsible for rendering the chat interface.
-->

<script lang="ts">
	import type { PageProps } from './$types';
	import Chat from '../../Chat.svelte';

	let { data }: PageProps = $props();

	let messages = $derived(data.messages || []);
	let ok = $derived(data.ok); // Check if the response is valid
	let conversationId = $derived(data.slug || '');
</script>

{#if ok && messages}
	<Chat {conversationId} {messages} />
{:else}
	<!-- Invalid conversation ID -->
	<p>Invalid conversation ID</p>
	<div class="error-container">
		<p>Failed to load conversation. Please check the ID or try again later.</p>
		{#if data.error}
			<p>Error: {data.error}</p>
		{/if}
	</div>
{/if}
