import 'dotenv/config';

export default {
  expo: {
    name: 'KisaanDost',
    slug: 'kisaandost',
    version: '1.0.0',
    orientation: 'portrait',
    icon: './assets/icon.png',
    userInterfaceStyle: 'light',
    ios: {
      supportsTablet: true,
    },
    android: {
      adaptiveIcon: {
        backgroundColor: '#E6F4FE',
        foregroundImage: './assets/android-icon-foreground.png',
        backgroundImage: './assets/android-icon-background.png',
        monochromeImage: './assets/android-icon-monochrome.png',
      },
      predictiveBackGestureEnabled: false,
    },
    web: {
      favicon: './assets/favicon.png',
    },
    plugins: [
      [
        'expo-audio',
        {
          microphonePermission: 'Allow KisaanDost to access your microphone for voice commands.',
        },
      ],
      'expo-localization',
      'expo-font',
      'expo-asset',
    ],
    extra: {
      /** API base URL for real backend calls. Sourced from process.env.API_BASE_URL (.env file) */
      apiBaseUrl: process.env.API_BASE_URL || 'http://localhost:8000',
      useMock: process.env.USE_MOCK === 'true',
    },
  },
};
