/**
 * LanguageContext — React context that manages the active UI language.
 *
 * ## Design Decisions
 *
 * 1. **Default locale is always Urdu (`ur`).**
 *    The target audience is Urdu/Punjabi-speaking farmers in Pakistan.
 *    Defaulting to English for non-Urdu device locales would be the wrong
 *    choice — even if the device is set to e.g. Arabic or French, the
 *    farmer using this app is far more likely to need Urdu.
 *
 * 2. **No RTL layout flip.**
 *    We intentionally do NOT call `I18nManager.forceRTL(true)` or
 *    `allowRTL(true)`.  The entire app layout stays LTR.  Urdu text is
 *    right-aligned at the *component* level using `textAlign: 'right'`
 *    and `writingDirection: 'rtl'` on individual `<Text>` elements.
 *    This avoids the massive layout breakage that a global RTL flip
 *    would cause (navigation chrome, marketplace cards, etc.) and keeps
 *    the UI predictable for low-literacy users.
 *    *** Do NOT "fix" this by enabling global RTL. ***
 *
 * 3. **Persistence via AsyncStorage.**
 *    The user's manual language toggle is stored under the key
 *    `kisaandost_language` so it survives app restarts.
 */
import React, {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
} from 'react';
import AsyncStorage from '@react-native-async-storage/async-storage';
import * as Localization from 'expo-localization';
import { i18n } from '../i18n';

const STORAGE_KEY = 'kisaandost_language';

type Language = 'ur' | 'en';

interface LanguageContextValue {
  /** Currently active language code. */
  language: Language;
  /** Switch the active language (persisted to AsyncStorage). */
  setLanguage: (lang: Language) => void;
  /** Toggle between Urdu and English (persisted to AsyncStorage). */
  toggleLanguage: () => void;
  /** Shorthand translate function bound to the current language. */
  t: (key: string, params?: Record<string, string | number>) => string;
}

const LanguageContext = createContext<LanguageContextValue>({
  language: 'ur',
  setLanguage: () => {},
  toggleLanguage: () => {},
  t: (key: string) => key,
});

/**
 * Detect the best initial language:
 *  1. Check AsyncStorage for a previously saved preference.
 *  2. Check the device locale — if it starts with "ur", use Urdu.
 *  3. Otherwise still default to Urdu (see design decision #1 above).
 */
async function detectInitialLanguage(): Promise<Language> {
  try {
    const stored = await AsyncStorage.getItem(STORAGE_KEY);
    if (stored === 'en' || stored === 'ur') {
      return stored;
    }
  } catch {
    // AsyncStorage read failed — fall through to locale detection.
  }

  // Check device locale
  const locales = Localization.getLocales();
  if (locales && locales.length > 0) {
    const deviceLang = locales[0]?.languageCode?.toLowerCase() ?? '';
    if (deviceLang.startsWith('ur')) {
      return 'ur';
    }
  }

  // Default to Urdu regardless of device locale (target audience decision).
  return 'ur';
}

export function LanguageProvider({ children }: { children: React.ReactNode }) {
  const [language, setLanguageState] = useState<Language>('ur');
  const [ready, setReady] = useState(false);

  // Bootstrap: detect stored / device language on mount.
  useEffect(() => {
    detectInitialLanguage().then((lang) => {
      i18n.locale = lang;
      setLanguageState(lang);
      setReady(true);
    });
  }, []);

  const setLanguage = useCallback((lang: Language) => {
    i18n.locale = lang;
    setLanguageState(lang);
    AsyncStorage.setItem(STORAGE_KEY, lang).catch(() => {
      // Silently ignore persistence failures.
    });
  }, []);

  const toggleLanguage = useCallback(() => {
    setLanguage(language === 'ur' ? 'en' : 'ur');
  }, [language, setLanguage]);

  /** Translate bound to the current language. */
  const t = useCallback(
    (key: string, params?: Record<string, string | number>) =>
      i18n.t(key, { ...(params ?? {}) }),
    // Re-create when language changes so consumers re-render.
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [language],
  );

  const value = useMemo(
    () => ({ language, setLanguage, toggleLanguage, t }),
    [language, setLanguage, toggleLanguage, t],
  );

  // Don't render children until we've resolved the initial language —
  // this prevents a flash of wrong-locale content.
  if (!ready) {
    return null;
  }

  return (
    <LanguageContext.Provider value={value}>
      {children}
    </LanguageContext.Provider>
  );
}

/**
 * Hook to access the current language, the setter, and the `t()` helper.
 *
 * ```tsx
 * const { t, language, setLanguage } = useLanguage();
 * ```
 */
export function useLanguage(): LanguageContextValue {
  return useContext(LanguageContext);
}
