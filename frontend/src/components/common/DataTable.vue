<script setup lang="ts">
import { ChevronUpIcon, ChevronDownIcon } from '@heroicons/vue/24/solid'

interface Column {
  key: string
  label: string
  sortable?: boolean
  align?: 'left' | 'center' | 'right'
  format?: (value: any, row: any) => string
}

defineProps<{
  columns: Column[]
  data: any[]
  sortBy?: string
  sortOrder?: 'asc' | 'desc'
  loading?: boolean
}>()

const emit = defineEmits<{
  sort: [column: string]
  rowClick: [row: any]
}>()

function getCellValue(row: any, column: Column): string {
  const value = row[column.key]
  if (column.format) {
    return column.format(value, row)
  }
  if (typeof value === 'number') {
    if (value >= 1000) {
      return `$${value.toLocaleString()}`
    }
    if (value < 1 && value > 0) {
      return `${(value * 100).toFixed(0)}%`
    }
    return value.toString()
  }
  return value ?? '-'
}

function handleSort(column: Column) {
  if (column.sortable) {
    emit('sort', column.key)
  }
}
</script>

<template>
  <div class="card-dark overflow-hidden">
    <div class="overflow-x-auto">
      <table class="min-w-full divide-y divide-surface-800">
        <thead class="bg-surface-800/50">
          <tr>
            <th
              v-for="column in columns"
              :key="column.key"
              :class="[
                'px-6 py-4 text-xs font-semibold text-surface-400 uppercase tracking-wider',
                column.align === 'right' ? 'text-right' : column.align === 'center' ? 'text-center' : 'text-left',
                column.sortable ? 'cursor-pointer hover:text-surface-200 select-none transition-colors' : '',
              ]"
              @click="handleSort(column)"
            >
              <div class="flex items-center gap-2" :class="column.align === 'right' ? 'justify-end' : ''">
                {{ column.label }}
                <span v-if="column.sortable && sortBy === column.key" class="text-accent-500">
                  <ChevronUpIcon v-if="sortOrder === 'asc'" class="w-4 h-4" />
                  <ChevronDownIcon v-else class="w-4 h-4" />
                </span>
              </div>
            </th>
          </tr>
        </thead>
        <tbody class="divide-y divide-surface-800/50">
          <tr v-if="loading">
            <td :colspan="columns.length" class="px-6 py-16 text-center">
              <div class="flex items-center justify-center gap-3">
                <div class="w-6 h-6 border-2 border-accent-500 rounded-full border-t-transparent animate-spin"></div>
                <span class="text-sm text-surface-400">Loading deals...</span>
              </div>
            </td>
          </tr>
          <tr v-else-if="data.length === 0">
            <td :colspan="columns.length" class="px-6 py-16 text-center">
              <p class="text-sm text-surface-500">No deals found matching your criteria</p>
            </td>
          </tr>
          <tr
            v-for="(row, index) in data"
            :key="index"
            class="hover:bg-surface-800/30 cursor-pointer transition-colors"
            @click="$emit('rowClick', row)"
          >
            <td
              v-for="column in columns"
              :key="column.key"
              :class="[
                'px-6 py-4 whitespace-nowrap text-sm',
                column.align === 'right' ? 'text-right' : column.align === 'center' ? 'text-center' : 'text-left',
              ]"
            >
              <slot :name="`cell-${column.key}`" :value="row[column.key]" :row="row">
                <span class="text-surface-200">{{ getCellValue(row, column) }}</span>
              </slot>
            </td>
          </tr>
        </tbody>
      </table>
    </div>
  </div>
</template>
