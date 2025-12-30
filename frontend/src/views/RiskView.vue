<script setup lang="ts">
import { computed } from 'vue'
import MetricCard from '@/components/common/MetricCard.vue'
import MultiSelect from '@/components/common/MultiSelect.vue'
import RangeSlider from '@/components/common/RangeSlider.vue'
import DealList from '@/components/deals/DealList.vue'
import { useDeals, useFilters as useFilterOptions, useSummary } from '@/composables/useDeals'
import { useFilterState } from '@/composables/useFilterState'
import {
  CurrencyDollarIcon,
  ExclamationTriangleIcon,
  ArrowTrendingUpIcon,
  TableCellsIcon,
  ArrowPathIcon,
  AdjustmentsHorizontalIcon,
} from '@heroicons/vue/24/solid'

const {
  account,
  salesAgent,
  product,
  riskCategory,
  minRiskScore,
  maxRiskScore,
  sortBy,
  sortOrder,
  filters,
  resetFilters,
} = useFilterState()

const { data: dealsData, isLoading: dealsLoading } = useDeals(filters)
const { data: filterOptions } = useFilterOptions()
const { data: summary } = useSummary()

const deals = computed(() => dealsData.value?.deals ?? [])

const riskCategories = computed(() => filterOptions.value?.risk_categories ?? [])
const accounts = computed(() => filterOptions.value?.accounts?.slice(0, 50) ?? [])
const salesAgents = computed(() => filterOptions.value?.sales_agents ?? [])
const products = computed(() => filterOptions.value?.products ?? [])

function handleSort(column: string) {
  if (sortBy.value === column) {
    sortOrder.value = sortOrder.value === 'asc' ? 'desc' : 'asc'
  } else {
    sortBy.value = column
    sortOrder.value = 'desc'
  }
}
</script>

<template>
  <div class="space-y-6">
    <!-- Header -->
    <div class="flex items-center justify-between animate-in">
      <div>
        <h1 class="text-3xl font-semibold text-surface-50 tracking-tight">Risk Analysis</h1>
        <p class="mt-2 text-surface-400">Deals sorted by risk and value priority</p>
      </div>
      <button
        @click="resetFilters"
        class="inline-flex items-center gap-2 px-4 py-2.5 bg-surface-800 hover:bg-surface-700 text-surface-200 rounded-xl border border-surface-700 transition-all text-sm font-medium"
      >
        <ArrowPathIcon class="w-4 h-4" />
        Reset Filters
      </button>
    </div>

    <!-- Summary Metrics -->
    <div v-if="summary" class="grid grid-cols-1 md:grid-cols-4 gap-5 animate-in stagger-1">
      <MetricCard
        label="At-Risk Revenue"
        :value="summary.at_risk_revenue"
        :icon="CurrencyDollarIcon"
        variant="danger"
      />
      <MetricCard
        label="High Risk Deals"
        :value="summary.high_risk_count"
        :icon="ExclamationTriangleIcon"
        variant="warning"
      />
      <MetricCard
        label="Avg Win Probability"
        :value="summary.avg_win_probability"
        :icon="ArrowTrendingUpIcon"
        variant="success"
      />
      <MetricCard
        label="Showing"
        :value="`${deals.length} of ${summary.total_deals}`"
        :icon="TableCellsIcon"
        variant="accent"
      />
    </div>

    <!-- Filters -->
    <div class="bg-surface-900 border border-surface-800 rounded-2xl shadow-card p-6 animate-in stagger-2 relative z-20" style="overflow: visible;">
      <div class="flex items-center gap-3 mb-5">
        <AdjustmentsHorizontalIcon class="w-5 h-5 text-accent-500" />
        <h3 class="text-sm font-semibold text-surface-200">Filters</h3>
      </div>
      <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-5">
        <!-- Risk Category -->
        <MultiSelect
          v-model="riskCategory"
          :options="riskCategories"
          label="Risk Level"
          placeholder="All levels"
        />

        <!-- Account -->
        <MultiSelect
          v-model="account"
          :options="accounts"
          label="Account"
          placeholder="All accounts"
        />

        <!-- Sales Rep -->
        <MultiSelect
          v-model="salesAgent"
          :options="salesAgents"
          label="Sales Rep"
          placeholder="All reps"
        />

        <!-- Product -->
        <MultiSelect
          v-model="product"
          :options="products"
          label="Product"
          placeholder="All products"
        />

        <!-- Risk Score Range -->
        <RangeSlider
          v-model:minValue="minRiskScore"
          v-model:maxValue="maxRiskScore"
          :min="0"
          :max="100"
          :step="1"
          label="Risk Score Range"
        />
      </div>
    </div>

    <!-- Deal Table -->
    <div class="animate-in stagger-3 relative z-10">
      <DealList
        :deals="deals"
        :loading="dealsLoading"
        :sort-by="sortBy"
        :sort-order="sortOrder"
        @sort="handleSort"
      />
    </div>
  </div>
</template>
