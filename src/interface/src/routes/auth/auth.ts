/* 
interface/src/api/auth.ts

This file contains the helper functions to interact with the user authentication API.
*/
const API_BASE_URL = "http://localhost:8080/auth"

interface UserData {
    username: string,
    email: string,
    password: string,
}

export async function postLogin(username: string, password: string): Promise<Response> {
    // Make a POST request to the login endpoint
    const loginData: UserData = {
        username: username,
        email: '', // Email is not required for login
        password: password
    }
    const response = await fetch(`${API_BASE_URL}/login`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(loginData),
        mode: 'cors',
        credentials: 'include',
    }).then((res) => {
        if (!res.ok) {
            throw new Error(`HTTP error! status: ${res.status}`);
        }
        return res.json();
    }).catch((error) => {
        console.error('Error during fetch:', error);
        throw error; // Re-throw the error for further handling
    })

    return response;
}

export async function postRegister(username: string, email: string, password: string): Promise<Response> {
    // Make a POST request to the registration endpoint
    const registerData: UserData = {
        username: username,
        email: email,
        password: password
    }
    const response = await fetch(`${API_BASE_URL}/register`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(registerData),
        mode: 'cors',
        credentials: 'include',
    }).then((res) => {
        if (!res.ok) {
            throw new Error(`HTTP error! status: ${res.status}`);
        }
        return res.json();
    }).catch((error) => {
        console.error('Error during fetch:', error);
        throw error; // Re-throw the error for further handling
    })

    return response;
}