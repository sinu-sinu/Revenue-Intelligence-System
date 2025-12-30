<script setup lang="ts">
import { computed } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import DealDetailComponent from '@/components/deals/DealDetail.vue'
import { useDeal } from '@/composables/useDeals'
import { ArrowLeftIcon } from '@heroicons/vue/24/solid'

const route = useRoute()
const router = useRouter()

const dealId = computed(() => route.params.id as string)
const { data: deal, isLoading, error } = useDeal(dealId)

function goBack() {
  router.back()
}
</script>

<template>
  <div class="space-y-6">
    <!-- Back Button -->
    <button
      @click="goBack"
      class="inline-flex items-center gap-2 text-surface-400 hover:text-surface-200 transition-colors text-sm font-medium animate-in"
    >
      <ArrowLeftIcon class="w-4 h-4" />
      <span>Back to Risk Dashboard</span>
    </button>

    <!-- Loading -->
    <div v-if="isLoading" class="flex items-center justify-center h-64">
      <div class="w-10 h-10 border-4 border-accent-500 rounded-full border-t-transparent animate-spin"></div>
    </div>

    <!-- Error -->
    <div v-else-if="error" class="card-dark p-6 border-rose-500/30">
      <h2 class="text-lg font-semibold text-rose-400">Error Loading Deal</h2>
      <p class="text-surface-400 mt-2 text-sm">{{ error.message }}</p>
    </div>

    <!-- Deal Detail -->
    <DealDetailComponent v-else-if="deal" :deal="deal" />
  </div>
</template>
