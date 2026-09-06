import React, { useEffect, useState, useRef, useCallback } from 'react';
import {
  StyleSheet,
  Text,
  View,
  TouchableOpacity,
  ActivityIndicator,
  Platform,
} from 'react-native';
import {
  useAudioRecorder,
  useAudioRecorderState,
  RecordingPresets,
  AudioModule,
  setAudioModeAsync,
  createAudioPlayer,
} from 'expo-audio';
import { useLanguage } from '../contexts/LanguageContext';
import { colors } from '../theme/colors';
import { spacing, radius } from '../theme/spacing';
import { fontSize, fontWeight } from '../theme/typography';

export interface VoiceRecorderProps {
  onRecordingStop?: (uri: string) => void;
  onCancel?: () => void;
  disabled?: boolean;
}

function formatTime(totalSeconds: number): string {
  const mins = Math.floor(totalSeconds / 60);
  const secs = totalSeconds % 60;
  return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
}

export default function VoiceRecorder({
  onRecordingStop,
  onCancel,
  disabled = false,
}: VoiceRecorderProps) {
  const { t } = useLanguage();
  const audioRecorder = useAudioRecorder(RecordingPresets.HIGH_QUALITY);
  const recorderState = useAudioRecorderState(audioRecorder);

  const [recordingUri, setRecordingUri] = useState<string | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [permissionGranted, setPermissionGranted] = useState<boolean | null>(null);
  const [isPreparing, setIsPreparing] = useState(false);
  const [elapsedSeconds, setElapsedSeconds] = useState(0);

  const playerRef = useRef<ReturnType<typeof createAudioPlayer> | null>(null);

  useEffect(() => {
    (async () => {
      const status = await AudioModule.requestRecordingPermissionsAsync();
      setPermissionGranted(status.granted);
      if (status.granted) {
        await setAudioModeAsync({ playsInSilentMode: true, allowsRecording: true });
      }
    })();
    return () => { playerRef.current?.remove(); };
  }, []);

  useEffect(() => {
    if (!recorderState.isRecording) {
      setElapsedSeconds(0);
      return;
    }
    const interval = setInterval(() => setElapsedSeconds((p) => p + 1), 1000);
    return () => clearInterval(interval);
  }, [recorderState.isRecording]);

  const startRecording = async () => {
    if (disabled) return;
    try {
      setIsPreparing(true);
      await audioRecorder.prepareToRecordAsync();
      audioRecorder.record();
    } catch (err) {
      console.warn('Failed to start recording:', err);
    } finally {
      setIsPreparing(false);
    }
  };

  const stopRecording = async () => {
    try {
      await audioRecorder.stop();
      const uri = audioRecorder.uri;
      if (uri) {
        playerRef.current?.remove();
        playerRef.current = createAudioPlayer(uri);
        setRecordingUri(uri);
        onRecordingStop?.(uri);
      }
    } catch (err) {
      console.warn('Failed to stop recording:', err);
    }
  };

  const cancelRecording = useCallback(async () => {
    try {
      await audioRecorder.stop();
      setRecordingUri(null);
      onCancel?.();
    } catch (err) {
      console.warn('Failed to cancel recording:', err);
    }
  }, [audioRecorder, onCancel]);

  const playRecording = () => {
    if (!playerRef.current) return;
    playerRef.current.seekTo(0);
    playerRef.current.play();
    setIsPlaying(true);
    const check = setInterval(() => {
      if (!playerRef.current) { clearInterval(check); setIsPlaying(false); return; }
      const { currentTime, duration } = playerRef.current;
      if (duration > 0 && currentTime >= duration - 0.1) {
        setIsPlaying(false);
        clearInterval(check);
      }
    }, 200);
  };

  if (permissionGranted === false) {
    return (
      <View style={styles.wrapper}>
        <Text style={styles.permIcon}>🔇</Text>
        <Text style={styles.permTitle}>{t('common.permissions.micDeniedTitle')}</Text>
        <Text style={styles.permMessage}>
          {t('common.permissions.micDeniedMessage')}
        </Text>
      </View>
    );
  }

  const isRecording = recorderState.isRecording;

  return (
    <View style={styles.wrapper}>
      {isRecording && (
        <View style={styles.recordingIndicator}>
          <View style={styles.redDot} />
          <Text style={styles.recordingText}>{t('common.recording.recording')}</Text>
          <Text style={styles.timerText}>{formatTime(elapsedSeconds)}</Text>
        </View>
      )}

      <TouchableOpacity
        style={[
          styles.micButton,
          isRecording && styles.micButtonRecording,
          disabled && styles.micButtonDisabled,
        ]}
        onPress={isRecording ? stopRecording : startRecording}
        disabled={isPreparing || disabled}
        activeOpacity={0.7}
      >
        {isPreparing ? (
          <ActivityIndicator color="#fff" size="large" />
        ) : isRecording ? (
          <Text style={styles.micIcon}>⏹</Text>
        ) : (
          <Text style={styles.micIcon}>🎙️</Text>
        )}
      </TouchableOpacity>

      <Text style={[styles.micLabel, disabled && styles.micLabelDisabled]}>
        {isPreparing ? t('common.recording.preparing') : isRecording ? t('common.recording.tapToStop') : disabled ? t('common.recording.pleaseWait') : t('common.recording.tapToRecord')}
      </Text>

      {isRecording && (
        <TouchableOpacity style={styles.cancelButton} onPress={cancelRecording}>
          <Text style={styles.cancelButtonText}>{t('common.recording.cancelButton')}</Text>
        </TouchableOpacity>
      )}

      {recordingUri && !isRecording && !disabled && (
        <TouchableOpacity
          style={[styles.playButton, isPlaying && styles.playButtonActive]}
          onPress={playRecording}
          disabled={isPlaying}
        >
          <Text style={styles.playButtonText}>
            {isPlaying ? t('common.recording.playing') : t('common.recording.playRecording')}
          </Text>
        </TouchableOpacity>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  wrapper: {
    alignItems: 'center',
    paddingVertical: spacing.lg,
    width: '100%',
  },
  recordingIndicator: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: spacing.base,
  },
  redDot: {
    width: 12,
    height: 12,
    borderRadius: 6,
    backgroundColor: colors.error,
    marginRight: spacing.sm,
  },
  recordingText: {
    fontSize: fontSize.body,
    fontWeight: fontWeight.semibold,
    color: colors.error,
  },
  timerText: {
    fontSize: fontSize.body,
    fontWeight: fontWeight.bold,
    color: colors.error,
    marginLeft: spacing.sm,
    fontVariant: ['tabular-nums'],
  },
  micButton: {
    width: 96,
    height: 96,
    borderRadius: 48,
    backgroundColor: colors.primary,
    justifyContent: 'center',
    alignItems: 'center',
    ...(Platform.OS === 'web'
      ? { boxShadow: '0 2px 4px rgba(0, 0, 0, 0.25)' }
      : {
          elevation: 4,
          shadowColor: '#000',
          shadowOffset: { width: 0, height: 2 },
          shadowOpacity: 0.25,
          shadowRadius: 4,
        }),
  },
  micButtonRecording: {
    backgroundColor: colors.error,
  },
  micButtonDisabled: {
    backgroundColor: colors.neutral,
    elevation: 0,
    ...(Platform.OS === 'web' ? { boxShadow: 'none' } : { shadowOpacity: 0 }),
  },
  micIcon: {
    fontSize: 40,
  },
  micLabel: {
    marginTop: spacing.md,
    fontSize: fontSize.body,
    fontWeight: fontWeight.semibold,
    color: colors.textSecondary,
  },
  micLabelDisabled: {
    color: colors.textMuted,
  },
  cancelButton: {
    marginTop: spacing.md,
    paddingHorizontal: spacing.base,
    paddingVertical: spacing.sm,
    borderRadius: radius.sm,
    backgroundColor: colors.background,
    borderWidth: 1,
    borderColor: colors.neutralBorder,
  },
  cancelButtonText: {
    fontSize: fontSize.sm,
    fontWeight: fontWeight.semibold,
    color: colors.textSecondary,
  },
  playButton: {
    marginTop: spacing.base,
    backgroundColor: colors.info,
    paddingHorizontal: spacing.lg,
    paddingVertical: spacing.md,
    borderRadius: radius.md,
    minWidth: 200,
    alignItems: 'center',
    elevation: 2,
  },
  playButtonActive: {
    backgroundColor: '#0d47a1',
  },
  playButtonText: {
    color: colors.textOnColor,
    fontSize: fontSize.body,
    fontWeight: fontWeight.semibold,
  },
  permIcon: {
    fontSize: 48,
    marginBottom: spacing.md,
  },
  permTitle: {
    fontSize: fontSize.heading,
    fontWeight: fontWeight.bold,
    color: colors.error,
    marginBottom: spacing.sm,
  },
  permMessage: {
    fontSize: fontSize.body,
    color: colors.textSecondary,
    textAlign: 'center',
    lineHeight: 24,
    paddingHorizontal: spacing.base,
  },
});
