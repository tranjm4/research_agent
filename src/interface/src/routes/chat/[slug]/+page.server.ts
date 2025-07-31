import type { PageServerLoad } from './$types';


const API_BASE_URL: string = 'http://localhost:8080';

interface Message {
    id: number;
    content: string;
    timestamp: Date;
    sender: 'user' | 'assistant' | 'error';
}

export const load: PageServerLoad = async ({ params, fetch }) => {
    const { slug } = params;
    // Get the conversation data based on the slug
    try {
        // console.log(`Fetching conversation data for slug: ${slug}`);
        const response = await fetch(`${API_BASE_URL}/chat/history/${slug}`, {
            method: 'GET',
            headers: {
                'Content-Type': 'application/json',
            },
            credentials: 'include',
            mode: 'cors',
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