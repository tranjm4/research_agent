<script module lang="ts">
	export interface User {
		name: string;
	}

	let user: User | null = $state(null);
	let isAuthenticated: boolean = $derived(!!user);

	export function createAuthStore() {
		return {
			get user() {
				return user;
			},
			get isAuthenticated() {
				return isAuthenticated;
			},
			setUser(newUser: User | null) {
				user = newUser;
			},
			logout() {
				user = null;
			}
		};
	}

	export const auth = createAuthStore();
</script>
