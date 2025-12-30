<script setup lang="ts">
import { ref, computed, onMounted, onUnmounted } from 'vue'
import { ChevronDownIcon, XMarkIcon, CheckIcon } from '@heroicons/vue/24/solid'

const props = defineProps<{
  modelValue: string[]
  options: string[]
  label: string
  placeholder?: string
}>()

const emit = defineEmits<{
  'update:modelValue': [value: string[]]
}>()

const isOpen = ref(false)
const dropdownRef = ref<HTMLDivElement>()

const selectedItems = computed({
  get: () => props.modelValue,
  set: (value) => emit('update:modelValue', value)
})

function toggleOption(option: string) {
  const index = selectedItems.value.indexOf(option)
  if (index === -1) {
    selectedItems.value = [...selectedItems.value, option]
  } else {
    selectedItems.value = selectedItems.value.filter(item => item !== option)
  }
}

function removeItem(option: string) {
  selectedItems.value = selectedItems.value.filter(item => item !== option)
}

function clearAll() {
  selectedItems.value = []
}

function isSelected(option: string) {
  return selectedItems.value.includes(option)
}

function handleClickOutside(event: MouseEvent) {
  if (dropdownRef.value && !dropdownRef.value.contains(event.target as Node)) {
    isOpen.value = false
  }
}

onMounted(() => {
  document.addEventListener('click', handleClickOutside)
})

onUnmounted(() => {
  document.removeEventListener('click', handleClickOutside)
})
</script>

<template>
  <div ref="dropdownRef" class="relative">
    <label class="block text-xs font-medium text-surface-400 mb-2">{{ label }}</label>

    <!-- Trigger Button -->
    <button
      type="button"
      @click="isOpen = !isOpen"
      class="w-full flex items-center justify-between gap-2 px-4 py-2.5 bg-surface-800/50 border border-surface-700 rounded-xl text-left text-sm text-surface-200 hover:border-surface-600 focus:outline-none focus:ring-2 focus:ring-accent-500/50 focus:border-accent-500 transition-all"
    >
      <span v-if="selectedItems.length === 0" class="text-surface-500">
        {{ placeholder || 'Select...' }}
      </span>
      <span v-else class="truncate">
        {{ selectedItems.length }} selected
      </span>
      <ChevronDownIcon
        :class="['w-4 h-4 text-surface-400 transition-transform', isOpen ? 'rotate-180' : '']"
      />
    </button>

    <!-- Selected Tags -->
    <div v-if="selectedItems.length > 0" class="flex flex-wrap gap-1.5 mt-2">
      <span
        v-for="item in selectedItems.slice(0, 3)"
        :key="item"
        class="inline-flex items-center gap-1 px-2 py-0.5 bg-accent-500/20 text-accent-400 text-xs rounded-lg"
      >
        {{ item }}
        <button @click.stop="removeItem(item)" class="hover:text-accent-300">
          <XMarkIcon class="w-3 h-3" />
        </button>
      </span>
      <span v-if="selectedItems.length > 3" class="text-xs text-surface-500 self-center">
        +{{ selectedItems.length - 3 }} more
      </span>
      <button
        v-if="selectedItems.length > 0"
        @click.stop="clearAll"
        class="text-xs text-surface-500 hover:text-surface-300 ml-1"
      >
        Clear
      </button>
    </div>

    <!-- Dropdown -->
    <div
      v-if="isOpen"
      class="absolute z-[100] w-full mt-2 bg-surface-800 border border-surface-700 rounded-xl shadow-lg overflow-hidden"
    >
      <div class="max-h-48 overflow-y-auto">
        <button
          v-for="option in options"
          :key="option"
          type="button"
          @click="toggleOption(option)"
          :class="[
            'w-full flex items-center gap-3 px-4 py-2.5 text-left text-sm transition-colors',
            isSelected(option)
              ? 'bg-accent-500/10 text-accent-400'
              : 'text-surface-300 hover:bg-surface-700/50'
          ]"
        >
          <div
            :class="[
              'w-4 h-4 rounded border flex items-center justify-center flex-shrink-0',
              isSelected(option)
                ? 'bg-accent-500 border-accent-500'
                : 'border-surface-600'
            ]"
          >
            <CheckIcon v-if="isSelected(option)" class="w-3 h-3 text-surface-950" />
          </div>
          <span class="truncate">{{ option }}</span>
        </button>
      </div>
    </div>
  </div>
</template>
