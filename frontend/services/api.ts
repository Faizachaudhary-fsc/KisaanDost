/**
 * KisaanDost Mock API Layer
 *
 * All functions are async and return Promises that resolve to response
 * objects. Expected errors come back as { success: false, error } in the
 * resolved value — they are never thrown. This mirrors the real backend's
 * response shape so the UI handles both paths identically.
 *
 * Set USE_MOCK=true in frontend/.env only for offline UI development. Real
 * mode routes each function through fetch() calls to API_BASE_URL.
 */

import Constants from 'expo-constants';
import { Platform } from 'react-native';

// Set USE_MOCK=true in frontend/.env only when developing without the backend.
const USE_MOCK = Constants.expoConfig?.extra?.useMock === true;

/**
 * Real backend base URL — sourced from app.config.js via Constants.expoConfig.extra.apiBaseUrl
 * EXPO_PUBLIC_API_BASE_URL is used for a physical device or tunnel.
 * Android emulator and web receive safe local defaults below.
 */
const configuredApiBaseUrl = String(Constants.expoConfig?.extra?.apiBaseUrl || '').replace(/\/$/, '');
const configuredIsLocalhost = /https?:\/\/(localhost|127\.0\.0\.1)(:\d+)?$/i.test(configuredApiBaseUrl);

export const API_BASE_URL: string =
  Platform.OS === 'web'
    ? (configuredApiBaseUrl && !configuredIsLocalhost ? configuredApiBaseUrl : 'http://localhost:8000')
    : (configuredApiBaseUrl || 'http://localhost:8000');

const VOICE_REQUEST_TIMEOUT_MS = 90_000;
const DEFAULT_REQUEST_TIMEOUT_MS = 15_000;

// Log the configured API URL during development (comment out in production)
if (__DEV__) {
  console.log('[api.ts] Configured API_BASE_URL:', API_BASE_URL);
}

// ────────────────────────────────────────────────────────────────────────────
// Types
// ────────────────────────────────────────────────────────────────────────────

export type Crop = 'wheat' | 'rice' | 'cotton' | 'maize';

export interface Listing {
  _id: string;
  farmerId: string;
  crop: Crop;
  quantity: number; // kg
  price: number; // PKR/kg
  location: string;
  phone: string; // 11 digits
  createdAt: string; // ISO timestamp
}

export interface ListingInput {
  farmerId?: string;
  crop?: Crop;
  quantity?: number;
  price?: number;
  location?: string;
  phone?: string;
}

// ── Voice endpoint response shapes ──────────────────────────────────────────

export interface VoiceHappyResponse {
  success: true;
  transcription: string;
  language: string;
  answer: string;
  audio_base64: string;
}

export interface VoiceUnrecognizedResponse {
  success: true;
  transcription: '';
  language: 'unrecognized';
  answer: string;
  audio_base64: string;
}

export type VoiceResponse = VoiceHappyResponse | VoiceUnrecognizedResponse;

// ── Generic response wrappers ───────────────────────────────────────────────

export interface SuccessResponse<T> {
  success: true;
  [key: string]: any;
}

export interface ErrorResponse {
  success: false;
  error: string;
}

export type ApiResponse<T = any> = SuccessResponse<T> | ErrorResponse;

// ────────────────────────────────────────────────────────────────────────────
// Helpers
// ────────────────────────────────────────────────────────────────────────────

/** Short random hex ID (good enough for mock data) */
function generateId(): string {
  return Math.random().toString(36).substring(2, 10) +
    Date.now().toString(36).substring(4);
}

/** Simulate network latency */
function delay(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function requestJson<T>(
  path: string,
  init?: RequestInit,
  timeoutMs: number = DEFAULT_REQUEST_TIMEOUT_MS,
): Promise<T | ErrorResponse> {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetch(`${API_BASE_URL}${path}`, { ...init, signal: controller.signal });
    let body: T | ErrorResponse;

    try {
      body = await response.json();
    } catch {
      throw new Error(`Backend returned an invalid response (${response.status})`);
    }

    if (!response.ok) {
      if (typeof body === 'object' && body !== null && 'success' in body && body.success === false) {
        return body as ErrorResponse;
      }
      throw new Error(`Backend request failed (${response.status})`);
    }

    return body;
  } catch (error) {
    if (error instanceof DOMException && error.name === 'AbortError') {
      throw new Error(timeoutMs === VOICE_REQUEST_TIMEOUT_MS ? 'Voice request timed out' : 'Backend request timed out');
    }
    throw error;
  } finally {
    clearTimeout(timeout);
  }
}

function audioFilename(mimeType: string, uri: string): string {
  const uriFilename = uri.split('/').pop()?.split('?')[0];
  if (uriFilename?.includes('.')) return uriFilename;

  const extensionByMimeType: Record<string, string> = {
    'audio/webm': 'webm',
    'audio/ogg': 'ogg',
    'audio/wav': 'wav',
    'audio/x-wav': 'wav',
    'audio/mp4': 'm4a',
    'audio/m4a': 'm4a',
    'audio/mpeg': 'mp3',
  };
  return `recording.${extensionByMimeType[mimeType.toLowerCase()] || 'm4a'}`;
}

function audioMimeType(uri: string): string {
  const extension = uri.split('?')[0].split('.').pop()?.toLowerCase();
  return ({
    webm: 'audio/webm',
    ogg: 'audio/ogg',
    wav: 'audio/wav',
    m4a: 'audio/m4a',
    mp4: 'audio/mp4',
    mp3: 'audio/mpeg',
  } as Record<string, string>)[extension || ''] || 'audio/m4a';
}

/**
 * Placeholder base64 audio string.
 * Represents a tiny silent .m4a — in production the backend returns real
 * TTS audio. This is a valid base64 structure but not a playable file.
 */
const PLACEHOLDER_AUDIO_BASE64 =
  'AAAAIGZ0eXBpc29tAAACAGlzb21pc28yYXZjMW1wNDEAAAAIZnJlZQAA';

// ────────────────────────────────────────────────────────────────────────────
// In-memory listings store (seeded with 3 samples)
// ────────────────────────────────────────────────────────────────────────────

const mockListings: Listing[] = [
  {
    _id: generateId(),
    farmerId: 'farmer_001',
    crop: 'wheat',
    quantity: 500,
    price: 3200,
    location: 'Lahore',
    phone: '03001234567',
    createdAt: new Date(Date.now() - 86400000 * 3).toISOString(), // 3 days ago
  },
  {
    _id: generateId(),
    farmerId: 'farmer_002',
    crop: 'rice',
    quantity: 200,
    price: 4500,
    location: 'Gujranwala',
    phone: '03212345678',
    createdAt: new Date(Date.now() - 86400000 * 1).toISOString(), // 1 day ago
  },
  {
    _id: generateId(),
    farmerId: 'farmer_003',
    crop: 'cotton',
    quantity: 1000,
    price: 2800,
    location: 'Multan',
    phone: '03334567890',
    createdAt: new Date().toISOString(), // today
  },
];

// ────────────────────────────────────────────────────────────────────────────
// Endpoint 1: sendVoiceQuery(audioUri, farmerId)
//
// Simulates: POST /api/assistant/voice
//            (multipart/form-data: audio file + optional farmerId)
//
// Mock: ~70% happy path, ~30% fallback (unrecognized).
// ────────────────────────────────────────────────────────────────────────────

/*
 * Error response shapes for reference (NOT triggered in mock):
 *
 * 400 — Bad Request:
 * {
 *   "success": false,
 *   "error": "No audio file provided"
 * }
 *
 * 500 — Internal Server Error:
 * {
 *   "success": false,
 *   "error": "ASR/LLM pipeline failed"
 * }
 */

export async function sendVoiceQuery(
  audioUri: string,
  farmerId?: string
): Promise<VoiceResponse | ErrorResponse> {
  if (!USE_MOCK) {
    const formData = new FormData();

    if (Platform.OS === 'web') {
      const audioResponse = await fetch(audioUri);
      if (!audioResponse.ok) {
        throw new Error(`Unable to read recorded audio (${audioResponse.status})`);
      }
      const audioBlob = await audioResponse.blob();
      const mimeType = audioBlob.type || 'audio/webm';
      formData.append('audio', audioBlob, audioFilename(mimeType, audioUri));
    } else {
      const mimeType = audioMimeType(audioUri);
      const filename = audioFilename(mimeType, audioUri);
      formData.append('audio', { uri: audioUri, type: mimeType, name: filename } as any);
    }

    if (farmerId) formData.append('farmerId', farmerId);
    return requestJson<VoiceResponse>('/api/assistant/voice', {
      method: 'POST',
      body: formData,
    }, VOICE_REQUEST_TIMEOUT_MS);
  }

  await delay(1500);

  const isHappy = Math.random() < 0.7;

  if (isHappy) {
    return {
      success: true,
      transcription: 'meri gandum de pattay peelay ne',
      language: 'urdu',
      answer:
        'Yeh nitrogen ki kami ho sakti hai. Urea spray kareen — 1 bori paani mein mix kar ke 1 acre pe chidakain.',
      audio_base64: PLACEHOLDER_AUDIO_BASE64,
    };
  }

  return {
    success: true,
    transcription: '',
    language: 'unrecognized',
    answer: 'Maazrat, samajh nahi aaya. Dobara Urdu mein poochain.',
    audio_base64: PLACEHOLDER_AUDIO_BASE64,
  };
}

// ────────────────────────────────────────────────────────────────────────────
// Endpoint 2: createListing(listing)
//
// Simulates: POST /api/listings
// Input: { farmerId, crop, quantity, price, location, phone }
// ────────────────────────────────────────────────────────────────────────────

export async function createListing(
  listing: ListingInput
): Promise<ApiResponse> {
  if (!USE_MOCK) {
    return requestJson('/api/listings', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(listing),
    });
  }

  await delay(400);

  // Client-side-style field validation — mirrors what the backend returns
  const requiredFields: (keyof ListingInput)[] = [
    'farmerId',
    'crop',
    'quantity',
    'price',
    'location',
    'phone',
  ];

  for (const field of requiredFields) {
    if (listing[field] === undefined || listing[field] === null || listing[field] === '') {
      return {
        success: false,
        error: `Missing required field: ${field}`,
      };
    }
  }

  const newListing: Listing = {
    _id: generateId(),
    farmerId: listing.farmerId!,
    crop: listing.crop!,
    quantity: listing.quantity!,
    price: listing.price!,
    location: listing.location!,
    phone: listing.phone!,
    createdAt: new Date().toISOString(),
  };

  mockListings.push(newListing);

  return {
    success: true,
    listing: newListing,
  };
}

// ────────────────────────────────────────────────────────────────────────────
// Endpoint 3: getListings(filters?)
//
// Simulates: GET /api/listings?crop=&location=
// Returns empty array as valid success — never treats empty as error.
// ────────────────────────────────────────────────────────────────────────────

export async function getListings(
  filters: { crop?: Crop; location?: string } = {}
): Promise<ApiResponse> {
  if (!USE_MOCK) {
    const params = new URLSearchParams();
    if (filters.crop) params.set('crop', filters.crop);
    if (filters.location?.trim()) params.set('location', filters.location.trim());
    const query = params.toString();
    return requestJson(`/api/listings${query ? `?${query}` : ''}`);
  }

  await delay(300);

  let results = [...mockListings];

  if (filters.crop) {
    results = results.filter((l) => l.crop === filters.crop);
  }

  if (filters.location) {
    const loc = filters.location.toLowerCase();
    results = results.filter((l) => l.location.toLowerCase().includes(loc));
  }

  return {
    success: true,
    listings: results,
  };
}

// ────────────────────────────────────────────────────────────────────────────
// Endpoint 4: getListingById(id)
//
// Simulates: GET /api/listings/:id
// ────────────────────────────────────────────────────────────────────────────

export async function getListingById(id: string): Promise<ApiResponse> {
  if (!USE_MOCK) {
    return requestJson(`/api/listings/${encodeURIComponent(id)}`);
  }

  await delay(200);

  const listing = mockListings.find((l) => l._id === id);

  if (!listing) {
    return { success: false, error: 'Listing not found' };
  }

  return { success: true, listing };
}

// ────────────────────────────────────────────────────────────────────────────
// Endpoint 5: deleteListing(id)
//
// Simulates: DELETE /api/listings/:id
// ────────────────────────────────────────────────────────────────────────────

export async function deleteListing(id: string): Promise<ApiResponse> {
  if (!USE_MOCK) {
    return requestJson(`/api/listings/${encodeURIComponent(id)}`, { method: 'DELETE' });
  }

  await delay(300);

  const index = mockListings.findIndex((l) => l._id === id);

  if (index === -1) {
    return { success: false, error: 'Listing not found' };
  }

  mockListings.splice(index, 1);

  return { success: true, message: 'Listing deleted' };
}
