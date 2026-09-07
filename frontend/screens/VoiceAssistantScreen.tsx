import React, { useState, useRef, useEffect, useCallback } from 'react';
import {
  StyleSheet,
  Text,
  View,
  ScrollView,
  ActivityIndicator,
} from 'react-native';
import { useNavigation } from '@react-navigation/native';
import { createAudioPlayer } from 'expo-audio';
import { Platform } from 'react-native';
import VoiceRecorder from '../components/VoiceRecorder';
import { StateCard, PrimaryButton } from '../components/ui';
import LanguageToggle from '../components/ui/LanguageToggle';
import { sendVoiceQuery, type VoiceResponse, type ApiResponse } from '../services/api';
import { decodeBase64Audio, deleteTempAudio } from '../utils/audio';
import { useFarmerContext } from '../contexts/FarmerContext';
import { useLanguage } from '../contexts/LanguageContext';
import { colors } from '../theme/colors';
import { spacing, radius } from '../theme/spacing';
import { fontSize, fontWeight, lineHeight, getFontFamily } from '../theme/typography';

export default function VoiceAssistantScreen() {
  const { farmerId } = useFarmerContext();
  const { language, t } = useLanguage();
  const navigation = useNavigation();
  const [isQuerying, setIsQuerying] = useState(false);
  const [response, setResponse] = useState<(VoiceResponse & { success: true }) | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [hasAudio, setHasAudio] = useState(false);

  const ttsPlayerRef = useRef<ReturnType<typeof createAudioPlayer> | null>(null);
  const webAudioRef = useRef<HTMLAudioElement | null>(null);
  const ttsFileUriRef = useRef<string | null>(null);

  useEffect(() => {
    return () => {
      ttsPlayerRef.current?.remove();
      webAudioRef.current?.pause();
      if (ttsFileUriRef.current) {
        deleteTempAudio(ttsFileUriRef.current);
      }
    };
  }, []);

  // Set header with LanguageToggle component
  useEffect(() => {
    navigation.setOptions({
      headerRight: () => <LanguageToggle />,
    });
  }, [navigation]);

  const resetState = useCallback(() => {
    setError(null);
    setResponse(null);
    setHasAudio(false);
    ttsPlayerRef.current?.remove();
    ttsPlayerRef.current = null;
    webAudioRef.current?.pause();
    webAudioRef.current = null;
    if (ttsFileUriRef.current) {
      deleteTempAudio(ttsFileUriRef.current);
      ttsFileUriRef.current = null;
    }
  }, []);

  const handleCancel = useCallback(() => { resetState(); }, [resetState]);
  const handleTryAgain = useCallback(() => { resetState(); }, [resetState]);

  const handleReplay = useCallback(() => {
    if (Platform.OS === 'web' && webAudioRef.current) {
      webAudioRef.current.currentTime = 0;
      void webAudioRef.current.play();
      return;
    }
    if (ttsPlayerRef.current) {
      ttsPlayerRef.current.seekTo(0);
      ttsPlayerRef.current.play();
    }
  }, []);

  const handleRecordingStop = async (audioUri: string) => {
    resetState();
    setIsQuerying(true);

    try {
      const result = await sendVoiceQuery(audioUri, farmerId ?? undefined);

      if ((result as ApiResponse).success === false) {
        setError((result as any).error ?? t('voice.errorFallback'));
        return;
      }

      const voiceResult = result as VoiceResponse;
      setResponse(voiceResult);

      if (voiceResult.audio_base64) {
        try {
          const fileUri = await decodeBase64Audio(voiceResult.audio_base64, '.wav');
          ttsFileUriRef.current = fileUri;
          if (Platform.OS === 'web') {
            const audio = new Audio(fileUri);
            webAudioRef.current = audio;
            void audio.play();
          } else {
            const player = createAudioPlayer(fileUri);
            ttsPlayerRef.current = player;
            player.play();
          }
          setHasAudio(true);
        } catch (audioErr) {
          console.warn('Failed to decode/play TTS audio:', audioErr);
        }
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Voice AI service is temporarily unavailable.');
      console.warn('Voice query failed:', err);
    } finally {
      setIsQuerying(false);
    }
  };

  const isIdle = !isQuerying && !response && !error;
  const isSending = isQuerying;
  const isUnrecognized = response?.language === 'unrecognized';
  const hasResponse = !!response && !isQuerying;
  const hasError = !!error && !isQuerying;

  return (
    <ScrollView contentContainerStyle={styles.container}>
      <Text style={[styles.title, { fontFamily: getFontFamily(language) }]}>
        {t('voice.title')}
      </Text>
      <Text
        style={[
          styles.subtitle,
          { fontFamily: getFontFamily(language), textAlign: language === 'ur' ? 'right' : 'center' },
        ]}
      >
        {t('voice.subtitle')}
      </Text>

      <VoiceRecorder
        onRecordingStop={handleRecordingStop}
        onCancel={handleCancel}
        disabled={isQuerying}
      />

      {/* ── Sending/Thinking ─────────────────────────────────────────── */}
      {isSending && (
        <View style={styles.sendingContainer}>
          <ActivityIndicator size="large" color={colors.accent} />
          <Text
            style={[
              styles.sendingText,
              { fontFamily: getFontFamily(language), textAlign: language === 'ur' ? 'right' : 'center' },
            ]}
          >
            {t('voice.thinking')}
          </Text>
        </View>
      )}

      {/* ── Error state ─────────────────────────────────────────────── */}
      {hasError && (
        <StateCard
          variant="error"
          icon="⚠️"
          title={t('voice.errorTitle')}
          description={error ?? undefined}
          actionLabel={t('voice.tryAgain')}
          onAction={handleTryAgain}
          language={language}
        />
      )}

      {/* ── Unrecognized language ───────────────────────────────────── */}
      {hasResponse && isUnrecognized && (
        <StateCard
          variant="info"
          icon="🤔"
          title={t('voice.unrecognizedTitle')}
          description={response.answer}
          actionLabel={t('voice.tryAgain')}
          onAction={handleTryAgain}
          language={language}
        />
      )}

      {/* ── Successful response ─────────────────────────────────────── */}
      {hasResponse && !isUnrecognized && (
        <View style={styles.responseContainer}>
          <View style={styles.badge}>
            <Text style={styles.badgeText}>
              {response.language.charAt(0).toUpperCase() + response.language.slice(1)}
            </Text>
          </View>

          {response.transcription ? (
            <View style={styles.transcriptionCard}>
              <Text
                style={[
                  styles.cardLabel,
                  { fontFamily: getFontFamily(language), textAlign: language === 'ur' ? 'right' : 'left' },
                ]}
              >
                {t('voice.youSaid')}
              </Text>
              <Text
                style={[
                  styles.transcriptionText,
                  { fontFamily: getFontFamily('ur'), textAlign: 'right', writingDirection: 'rtl' },
                ]}
              >
                {response.transcription}
              </Text>
            </View>
          ) : null}

          <View style={styles.answerCard}>
            <Text
              style={[
                styles.cardLabel,
                { fontFamily: getFontFamily(language), textAlign: language === 'ur' ? 'right' : 'left' },
              ]}
            >
              {t('voice.kisaanSays')}
            </Text>
            <Text
              style={[
                styles.answerText,
                { fontFamily: getFontFamily('ur'), textAlign: 'right', writingDirection: 'rtl' },
              ]}
            >
              {response.answer}
            </Text>
          </View>

          {hasAudio && (
            <PrimaryButton
              label={t('voice.replayAnswer')}
              icon="🔊"
              variant="secondary"
              onPress={handleReplay}
              style={styles.actionBtn}
            />
          )}

          <PrimaryButton
            label={t('voice.askAnotherQuestion')}
            variant="primary"
            onPress={handleTryAgain}
            style={styles.actionBtn}
          />
        </View>
      )}

      {/* ── Idle state ──────────────────────────────────────────────── */}
      {isIdle && (
        <StateCard
          variant="neutral"
          icon="🌱"
          title={t('voice.idleTitle')}
          description={t('voice.idleDescription')}
          language={language}
        />
      )}
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flexGrow: 1,
    backgroundColor: colors.background,
    alignItems: 'center',
    padding: spacing.lg,
  },
  title: {
    fontSize: fontSize.hero,
    fontWeight: fontWeight.bold,
    color: colors.primary,
    marginBottom: spacing.sm,
  },
  subtitle: {
    fontSize: fontSize.body,
    color: colors.textSecondary,
    textAlign: 'center',
    lineHeight: lineHeight.normal,
    marginBottom: spacing.sm,
  },

  // Sending
  sendingContainer: {
    alignItems: 'center',
    marginTop: spacing.lg,
    padding: spacing.lg,
    backgroundColor: colors.accentLight,
    borderRadius: radius.lg,
    width: '100%',
  },
  sendingText: {
    marginTop: spacing.md,
    fontSize: fontSize.body,
    color: colors.accentDark,
    fontWeight: fontWeight.semibold,
  },

  // Response
  responseContainer: {
    marginTop: spacing.lg,
    width: '100%',
    alignItems: 'center',
  },
  badge: {
    paddingHorizontal: spacing.base,
    paddingVertical: 6,
    borderRadius: radius.xl,
    backgroundColor: colors.successLight,
    marginBottom: spacing.base,
  },
  badgeText: {
    fontSize: fontSize.sm,
    fontWeight: fontWeight.bold,
    color: colors.primary,
  },
  transcriptionCard: {
    width: '100%',
    backgroundColor: colors.surface,
    borderRadius: radius.base,
    padding: spacing.base,
    marginBottom: spacing.md,
    elevation: 1,
    borderLeftWidth: 4,
    borderLeftColor: colors.neutral,
  },
  answerCard: {
    width: '100%',
    backgroundColor: colors.surface,
    borderRadius: radius.base,
    padding: spacing.base,
    marginBottom: spacing.base,
    elevation: 2,
    borderLeftWidth: 4,
    borderLeftColor: colors.primary,
  },
  cardLabel: {
    fontSize: fontSize.sm,
    fontWeight: fontWeight.bold,
    color: colors.textSecondary,
    textTransform: 'uppercase',
    letterSpacing: 0.5,
    marginBottom: spacing.sm,
  },
  transcriptionText: {
    fontSize: fontSize.body,
    color: colors.textPrimary,
    lineHeight: lineHeight.normal,
    fontStyle: 'italic',
  },
  answerText: {
    fontSize: fontSize.bodyLg,
    color: colors.primaryDark,
    lineHeight: lineHeight.relaxed,
    fontWeight: fontWeight.medium,
  },
  actionBtn: {
    marginBottom: spacing.md,
    minWidth: 200,
  },
});
