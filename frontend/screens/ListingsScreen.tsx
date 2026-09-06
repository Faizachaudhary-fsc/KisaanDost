import React, { useState, useCallback, useEffect } from 'react';
import {
  StyleSheet,
  Text,
  View,
  FlatList,
  TextInput,
  TouchableOpacity,
  ActivityIndicator,
  RefreshControl,
  Pressable,
  Alert,
} from 'react-native';
import { Picker } from '@react-native-picker/picker';
import { useIsFocused, useNavigation } from '@react-navigation/native';
import type { NativeStackScreenProps } from '@react-navigation/native-stack';
import { getListings, deleteListing } from '../services/api';
import type { Listing, Crop } from '../services/api';
import { EmptyState } from '../components/ui';
import LanguageToggle from '../components/ui/LanguageToggle';
import { getCropIcon, getCropLabelI18n } from '../theme/crops';
import { useLanguage } from '../contexts/LanguageContext';
import { colors } from '../theme/colors';
import { spacing, radius } from '../theme/spacing';
import { fontSize, fontWeight, lineHeight, getFontFamily } from '../theme/typography';

// ── Navigation param list (shared across Marketplace screens) ───────────────
export type MarketplaceStackParamList = {
  ListingsScreen: undefined;
  ListingDetailScreen: { listingId: string };
  AddListingScreen: {
    editingListingId?: string;
    initialData?: {
      crop: Crop;
      quantity: number;
      price: number;
      location: string;
      phone: string;
    };
  };
};

type Props = NativeStackScreenProps<MarketplaceStackParamList, 'ListingsScreen'>;

export default function ListingsScreen({ navigation }: Props) {
  const isFocused = useIsFocused();
  const nav = useNavigation();
  const { language, t } = useLanguage();

  const [listings, setListings] = useState<Listing[]>([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [cropFilter, setCropFilter] = useState<Crop | ''>('');
  const [locationFilter, setLocationFilter] = useState('');
  const [deletingId, setDeletingId] = useState<string | null>(null);

  const fetchListings = useCallback(async (showSpinner = false) => {
    if (showSpinner) setLoading(true);
    try {
      const filters: { crop?: Crop; location?: string } = {};
      if (cropFilter) filters.crop = cropFilter;
      if (locationFilter.trim()) filters.location = locationFilter.trim();

      const result = await getListings(filters);
      if (result.success) {
        setListings((result as any).listings ?? []);
      }
    } catch (err) {
      console.warn('Failed to fetch listings:', err);
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, [cropFilter, locationFilter]);

  // Set header with LanguageToggle component
  useEffect(() => {
    nav.setOptions({
      headerRight: () => <LanguageToggle />,
    });
  }, [nav]);

  useEffect(() => {
    if (isFocused) {
      fetchListings(!refreshing);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isFocused, cropFilter, locationFilter]);

  const handleRefresh = () => {
    setRefreshing(true);
    fetchListings(false);
  };

  // ── Delete with confirmation ─────────────────────────────────────────────
  const handleDelete = (id: string) => {
    Alert.alert(
      t('marketplace.listingsScreen.deleteConfirmTitle'),
      t('marketplace.listingsScreen.deleteConfirmMessage'),
      [
        { text: t('marketplace.listingsScreen.cancelButton'), style: 'cancel' },
        {
          text: t('marketplace.listingsScreen.deleteButton'),
          style: 'destructive',
          onPress: async () => {
            setDeletingId(id);
            try {
              const result = await deleteListing(id);
              if (result.success || (!result.success && (result as any).error === 'Listing not found')) {
                await fetchListings(false);
              }
            } catch (err) {
              console.warn('Delete failed:', err);
            } finally {
              setDeletingId(null);
            }
          },
        },
      ]
    );
  };

  // ── Render a single listing card ────────────────────────────────────────
  const renderItem = ({ item }: { item: Listing }) => (
    <Pressable
      style={styles.card}
      onPress={() => navigation.navigate('ListingDetailScreen', { listingId: item._id })}
    >
      <View style={styles.cardHeader}>
        <View style={styles.cropBadge}>
          <Text style={[styles.cropBadgeText, { fontFamily: getFontFamily(language) }]}>
            {getCropIcon(item.crop)} {getCropLabelI18n(item.crop, t)}
          </Text>
        </View>
        <Pressable
          style={styles.deleteBtn}
          onPress={() => handleDelete(item._id)}
          disabled={deletingId === item._id}
          hitSlop={8}
        >
          {deletingId === item._id ? (
            <ActivityIndicator size="small" color={colors.error} />
          ) : (
            <Text style={styles.deleteBtnText}>✕</Text>
          )}
        </Pressable>
      </View>

      <View style={styles.cardRow}>
        <Text style={[styles.label, { fontFamily: getFontFamily(language) }]}>
          {t('marketplace.listingsScreen.quantity')}
        </Text>
        <Text style={[styles.value, { fontFamily: getFontFamily(language) }]}>
          {item.quantity.toLocaleString()} kg
        </Text>
      </View>
      <View style={styles.cardRow}>
        <Text style={[styles.label, { fontFamily: getFontFamily(language) }]}>
          {t('marketplace.listingsScreen.price')}
        </Text>
        <Text style={[styles.value, { fontFamily: getFontFamily(language) }]}>
          PKR {item.price.toLocaleString()}/kg
        </Text>
      </View>
      <View style={styles.cardRow}>
        <Text style={[styles.label, { fontFamily: getFontFamily(language) }]}>
          {t('marketplace.listingsScreen.location')}
        </Text>
        <Text style={[styles.value, { fontFamily: getFontFamily(language) }]}>
          {item.location}
        </Text>
      </View>
      <View style={styles.cardRow}>
        <Text style={[styles.label, { fontFamily: getFontFamily(language) }]}>
          {t('marketplace.listingsScreen.phone')}
        </Text>
        <Text style={[styles.value, { fontFamily: getFontFamily(language) }]}>
          {item.phone}
        </Text>
      </View>
    </Pressable>
  );

  // ── Filter header ───────────────────────────────────────────────────────
  const CROP_FILTER_OPTIONS: { label: string; value: Crop | '' }[] = [
    { label: t('crops.allCrops'), value: '' },
    { label: `${getCropIcon('wheat')} ${t('crops.wheat.labelFull')}`, value: 'wheat' },
    { label: `${getCropIcon('rice')} ${t('crops.rice.labelFull')}`, value: 'rice' },
    { label: `${getCropIcon('cotton')} ${t('crops.cotton.labelFull')}`, value: 'cotton' },
    { label: `${getCropIcon('maize')} ${t('crops.maize.labelFull')}`, value: 'maize' },
  ];

  const ListHeader = (
    <View style={styles.filterContainer}>
      <Text style={[styles.filterLabel, { fontFamily: getFontFamily(language) }]}>
        {t('marketplace.listingsScreen.filterByCrop')}
      </Text>
      <View style={styles.pickerWrapper}>
        <Picker
          selectedValue={cropFilter}
          onValueChange={(val) => setCropFilter(val as Crop | '')}
          style={styles.picker}
          dropdownIconColor={colors.primary}
        >
          {CROP_FILTER_OPTIONS.map((opt) => (
            <Picker.Item key={opt.value} label={opt.label} value={opt.value} />
          ))}
        </Picker>
      </View>

      <Text style={[styles.filterLabel, { fontFamily: getFontFamily(language) }]}>
        {t('marketplace.listingsScreen.filterByLocation')}
      </Text>
      <TextInput
        style={[styles.textInput, { fontFamily: getFontFamily(language) }]}
        placeholder={t('marketplace.listingsScreen.locationPlaceholder')}
        placeholderTextColor="#aaa"
        value={locationFilter}
        onChangeText={setLocationFilter}
        autoCorrect={false}
      />
    </View>
  );

  // ── Empty state ─────────────────────────────────────────────────────────
  const ListEmpty = loading ? null : (
    <EmptyState
      icon="🌾"
      title={t('marketplace.listingsScreen.noListingsYet')}
      description={
        cropFilter || locationFilter
          ? t('marketplace.listingsScreen.noResultsMatchFilters')
          : t('marketplace.listingsScreen.beFirstListing')
      }
      language={language}
    />
  );

  return (
    <View style={styles.container}>
      {loading && listings.length === 0 ? (
        <View style={styles.initialLoader}>
          <ActivityIndicator size="large" color={colors.primary} />
          <Text style={[styles.loadingText, { fontFamily: getFontFamily(language) }]}>
            {t('marketplace.listingsScreen.loadingListings')}
          </Text>
        </View>
      ) : (
        <FlatList
          data={listings}
          keyExtractor={(item) => item._id}
          renderItem={renderItem}
          ListHeaderComponent={ListHeader}
          ListEmptyComponent={ListEmpty}
          contentContainerStyle={styles.listContent}
          refreshControl={
            <RefreshControl
              refreshing={refreshing}
              onRefresh={handleRefresh}
              tintColor={colors.primary}
              colors={[colors.primary]}
            />
          }
        />
      )}

      <TouchableOpacity
        style={styles.fab}
        onPress={() => navigation.navigate('AddListingScreen', {})}
        activeOpacity={0.85}
      >
        <Text style={styles.fabText}>＋</Text>
      </TouchableOpacity>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: colors.background,
  },
  initialLoader: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
  },
  loadingText: {
    marginTop: spacing.md,
    fontSize: fontSize.body,
    color: colors.primary,
    fontWeight: fontWeight.medium,
  },
  listContent: {
    paddingHorizontal: spacing.base,
    paddingBottom: 100,
  },

  // Filters
  filterContainer: {
    paddingTop: spacing.base,
    paddingBottom: spacing.sm,
  },
  filterLabel: {
    fontSize: fontSize.sm,
    fontWeight: fontWeight.bold,
    color: colors.textSecondary,
    textTransform: 'uppercase',
    letterSpacing: 0.5,
    marginBottom: spacing.xs,
    marginTop: spacing.sm,
  },
  pickerWrapper: {
    backgroundColor: colors.surface,
    borderRadius: radius.md,
    borderWidth: 1,
    borderColor: colors.border,
    overflow: 'hidden',
    marginBottom: spacing.xs,
  },
  picker: {
    color: colors.textPrimary,
  },
  textInput: {
    backgroundColor: colors.surface,
    borderRadius: radius.md,
    borderWidth: 1,
    borderColor: colors.border,
    paddingHorizontal: 14,
    paddingVertical: 10,
    fontSize: fontSize.body,
    color: colors.textPrimary,
    marginBottom: spacing.xs,
  },

  // Listing card
  card: {
    backgroundColor: colors.surface,
    borderRadius: radius.base,
    padding: spacing.base,
    marginTop: spacing.md,
    elevation: 2,
  },
  cardHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: spacing.md,
  },
  cropBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: colors.primaryLight,
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.xs,
    borderRadius: radius.xl,
  },
  cropBadgeText: {
    fontSize: fontSize.body,
    fontWeight: fontWeight.bold,
    color: colors.primary,
  },
  deleteBtn: {
    padding: 6,
    minWidth: 32,
    alignItems: 'center',
  },
  deleteBtnText: {
    fontSize: fontSize.body,
    color: colors.error,
    fontWeight: fontWeight.bold,
  },
  cardRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    paddingVertical: spacing.xs,
  },
  label: {
    fontSize: fontSize.body,
    color: colors.textMuted,
    fontWeight: fontWeight.medium,
  },
  value: {
    fontSize: fontSize.body,
    color: colors.textPrimary,
    fontWeight: fontWeight.semibold,
  },

  // FAB
  fab: {
    position: 'absolute',
    right: spacing.lg,
    bottom: 28,
    width: 60,
    height: 60,
    borderRadius: 30,
    backgroundColor: colors.primary,
    alignItems: 'center',
    justifyContent: 'center',
    elevation: 6,
  },
  fabText: {
    fontSize: 30,
    color: colors.textOnColor,
    lineHeight: 34,
  },
});
