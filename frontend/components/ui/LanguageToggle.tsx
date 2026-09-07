/**
 * LanguageToggle — simple pill / segmented control for switching between
 * Urdu (اردو) and English.
 *
 * Reads the current language from `LanguageContext` and calls `setLanguage`
 * when the user taps the opposite option.  Not yet placed in any screen —
 * it will be wired into a header / settings location in a later prompt.
 */
import React from 'react';
import { Pressable, StyleSheet, Text, View } from 'react-native';
import { useLanguage } from '../../contexts/LanguageContext';
import { colors } from '../../theme/colors';

export default function LanguageToggle() {
  const { language, toggleLanguage, t } = useLanguage();

  return (
    <View style={styles.container}>
      <Pressable
        onPress={toggleLanguage}
        style={[styles.pill, styles.pillActive]}
        accessibilityRole="button"
        accessibilityLabel={language === 'ur' ? t('common.language.english') : 'اردو'}
      >
        <Text style={[styles.label, styles.labelActive]}>
          {language === 'ur' ? t('common.language.english') : 'اردو'}
        </Text>
      </Pressable>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flexDirection: 'row',
    borderRadius: 20,
    borderWidth: 1,
    borderColor: colors.primary,
    overflow: 'hidden',
  },
  pill: {
    paddingHorizontal: 16,
    paddingVertical: 6,
  },
  pillActive: {
    backgroundColor: colors.primary,
  },
  label: {
    fontSize: 14,
    color: colors.primary,
    fontWeight: '600',
  },
  labelActive: {
    color: colors.textOnColor,
  },
});
