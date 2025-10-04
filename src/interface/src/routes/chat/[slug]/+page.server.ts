import type { PageServerLoad } from './$types';
import { env } from '$env/dynamic/private';

const API_BASE_URL: string = env.SERVER_BASE_URL + ':' + env.SERVER_PORT || 'http://localhost:8080';

interface Message {
    id: number;
    content: string;
    timestamp: Date;
    sender: 'user' | 'assistant' | 'error';
}

export const load: PageServerLoad = async ({ params, fetch, cookies }) => {
    const { slug } = params;
    // Get the conversation data based on the slug
    try {
        // Get JWT token from cookies - check both possible cookie names
        const sessionToken = cookies.get('session_token') || cookies.get('token');

        const headers: Record<string, string> = {
            'Content-Type': 'application/json',
        };

        // Include JWT token in Cookie header if available
        if (sessionToken) {
            headers['Cookie'] = `session_token=${sessionToken}`;
        }

        const response = await fetch(`${API_BASE_URL}/chat/history/${slug}`, {
            method: 'GET',
            headers,
            credentials: 'include',
        })

        if (!response.ok) {
            throw new Error(`Error fetching conversation: ${response.statusText}`);
        }

        const data = await response.json();
        // messages is a list of JSON objects

        let messages: Message[] = data.messages || [];
        messages = messages.map((msg: Message) => ({
            ...msg,
            isStreaming: false,
        }))
        let ok: boolean = data.ok || false;

        return {
            messages: messages,
            ok: ok,
            slug: params.slug,
            error: null,
        };
    } catch (error) {
        console.error('Error loading conversation:', error);
        return {
            response: null,
            ok: false,
            error: 'Failed to fetch conversation data.',
        }
    }
}