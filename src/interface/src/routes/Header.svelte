<script lang="ts">
	import { page } from '$app/state';
</script>

<header>
	<nav>
		<ul>
			<li aria-current={page.url.pathname === '/' ? 'page' : undefined}>
				<a href="/">Home</a>
			</li>
			<li aria-current={page.url.pathname === '/about' ? 'page' : undefined}>
				<a href="/about">About</a>
			</li>
			<li aria-current={page.url.pathname === '/auth/login' ? 'page' : undefined}>
				<a href="/auth/login">Login</a>
			</li>
			<li aria-current={page.url.pathname === '/auth/signup' ? 'page' : undefined}>
				<a href="/auth/signup">Sign Up</a>
			</li>
			<li aria-current={page.url.pathname === '/chat' ? 'page' : undefined}>
				<a href="/chat">Chat</a>
			</li>
		</ul>
	</nav>
</header>

<style>
	header {
		background: #333;
		color: white;
		padding: 1rem;
	}

	nav ul {
		list-style: none;
		padding: 0;
	}

	nav li {
		display: inline;
		margin-right: 1rem;
	}

	nav a {
		color: white;
		text-decoration: none;
	}

	nav a:hover {
		text-decoration: underline;
	}
</style>
