/**
 * Audio utility helpers.
 *
 * Provides reusable functions for working with base64-encoded audio data
 * — decoding it into temporary local files that expo-audio can play.
 */

import { Platform } from 'react-native';
import type { File as ExpoFile, Paths as ExpoPaths } from 'expo-file-system';

/**
 * Decodes a base64-encoded audio string into a temporary local .m4a file
 * and returns the file URI suitable for expo-audio's createAudioPlayer().
 *
 * Each call writes to a uniquely-named file to avoid collisions when
 * multiple audio clips are decoded in quick succession.
 *
 * @param base64 - The base64-encoded audio data (no data-URI prefix).
 * @param extension - File extension, defaults to '.m4a'.
 * @returns A local file URI (e.g. "file:///.../cache/tts_abc123.m4a").
 */
export async function decodeBase64Audio(
  base64: string,
  extension: string = '.m4a'
): Promise<string> {
  if (Platform.OS === 'web') {
    const binary = atob(base64);
    const bytes = Uint8Array.from(binary, (character) => character.charCodeAt(0));
    return URL.createObjectURL(new Blob([bytes], { type: 'audio/wav' }));
  }

  const { File, Paths } = require('expo-file-system') as {
    File: typeof ExpoFile;
    Paths: typeof ExpoPaths;
  };
  const fileName = `tts_${Date.now()}_${Math.random().toString(36).substring(2, 8)}${extension}`;
  const file = new File(Paths.cache, fileName);

  // Write the base64 data decoded as binary into the file
  file.write(base64, { encoding: 'base64' });

  return file.uri;
}

/**
 * Deletes a previously decoded temporary audio file.
 * Safe to call even if the file no longer exists.
 *
 * @param fileUri - The local file URI returned by decodeBase64Audio().
 */
export async function deleteTempAudio(fileUri: string): Promise<void> {
  if (Platform.OS === 'web') {
    URL.revokeObjectURL(fileUri);
    return;
  }

  try {
    const { File } = require('expo-file-system') as { File: typeof ExpoFile };
    const file = new File(fileUri);
    if (file.exists) {
      file.delete();
    }
  } catch {
    // File already gone or inaccessible — nothing to do
  }
}
