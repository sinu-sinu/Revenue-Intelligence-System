<script setup lang="ts">
import { ref, computed, watch } from 'vue'

const props = defineProps<{
  minValue: number
  maxValue: number
  min?: number
  max?: number
  step?: number
  label?: string
}>()

const emit = defineEmits<{
  'update:minValue': [value: number]
  'update:maxValue': [value: number]
}>()

const minLimit = computed(() => props.min ?? 0)
const maxLimit = computed(() => props.max ?? 100)
const stepValue = computed(() => props.step ?? 1)

// Local state to avoid twitching
const localMin = ref(props.minValue)
const localMax = ref(props.maxValue)

// Sync local state with props
watch(() => props.minValue, (val) => { localMin.value = val })
watch(() => props.maxValue, (val) => { localMax.value = val })

function onMinInput(event: Event) {
  const target = event.target as HTMLInputElement
  let value = Number(target.value)
  if (value >= localMax.value) {
    value = localMax.value - stepValue.value
  }
  localMin.value = value
  emit('update:minValue', value)
}

function onMaxInput(event: Event) {
  const target = event.target as HTMLInputElement
  let value = Number(target.value)
  if (value <= localMin.value) {
    value = localMin.value + stepValue.value
  }
  localMax.value = value
  emit('update:maxValue', value)
}

const rangeStyle = computed(() => {
  const minPercent = ((localMin.value - minLimit.value) / (maxLimit.value - minLimit.value)) * 100
  const maxPercent = ((localMax.value - minLimit.value) / (maxLimit.value - minLimit.value)) * 100
  return {
    left: `${minPercent}%`,
    width: `${maxPercent - minPercent}%`
  }
})
</script>

<template>
  <div>
    <label v-if="label" class="block text-xs font-medium text-surface-400 mb-2">{{ label }}</label>
    <!-- Match MultiSelect button height: py-2.5 = 10px top + 10px bottom, total ~42px -->
    <div class="h-[42px] flex items-center gap-3 px-4 bg-surface-800/50 border border-surface-700 rounded-xl">
      <!-- Min Value -->
      <span class="text-sm font-mono text-accent-400 w-8 text-center flex-shrink-0">{{ localMin }}</span>

      <!-- Dual Range Slider -->
      <div class="range-slider flex-1">
        <div class="range-track"></div>
        <div class="range-selected" :style="rangeStyle"></div>
        <input
          type="range"
          :min="minLimit"
          :max="maxLimit"
          :step="stepValue"
          :value="localMin"
          @input="onMinInput"
          class="range-input"
        />
        <input
          type="range"
          :min="minLimit"
          :max="maxLimit"
          :step="stepValue"
          :value="localMax"
          @input="onMaxInput"
          class="range-input"
        />
      </div>

      <!-- Max Value -->
      <span class="text-sm font-mono text-accent-400 w-8 text-center flex-shrink-0">{{ localMax }}</span>
    </div>
  </div>
</template>

<style scoped>
.range-slider {
  position: relative;
  height: 6px;
}

.range-track {
  position: absolute;
  width: 100%;
  height: 6px;
  background: #334155;
  border-radius: 3px;
}

.range-selected {
  position: absolute;
  height: 6px;
  background: #14b8a6;
  border-radius: 3px;
}

.range-input {
  position: absolute;
  width: 100%;
  height: 6px;
  -webkit-appearance: none;
  appearance: none;
  background: transparent;
  pointer-events: none;
  margin: 0;
  padding: 0;
}

.range-input::-webkit-slider-thumb {
  -webkit-appearance: none;
  appearance: none;
  width: 16px;
  height: 16px;
  background: #2dd4bf;
  border-radius: 50%;
  cursor: pointer;
  pointer-events: auto;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
  transition: transform 0.15s ease, background 0.15s ease;
}

.range-input::-webkit-slider-thumb:hover {
  background: #5eead4;
  transform: scale(1.1);
}

.range-input::-webkit-slider-thumb:active {
  transform: scale(1.15);
}

.range-input::-moz-range-thumb {
  width: 16px;
  height: 16px;
  background: #2dd4bf;
  border: none;
  border-radius: 50%;
  cursor: pointer;
  pointer-events: auto;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
  transition: transform 0.15s ease, background 0.15s ease;
}

.range-input::-moz-range-thumb:hover {
  background: #5eead4;
  transform: scale(1.1);
}
</style>
