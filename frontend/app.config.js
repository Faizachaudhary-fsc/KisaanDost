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
      /** Optional LAN/tunnel URL for physical devices; platform defaults are selected in api.ts. */
      apiBaseUrl: process.env.EXPO_PUBLIC_API_BASE_URL || process.env.API_BASE_URL || '',
      useMock: process.env.USE_MOCK === 'true',
    },
  },
};
