<script setup lang="ts">
import { ref } from 'vue'
import { RouterLink, RouterView, useRoute } from 'vue-router'
import {
  ChartBarIcon,
  ShieldExclamationIcon,
  ChartPieIcon,
  ChevronLeftIcon,
  ChevronRightIcon,
} from '@heroicons/vue/24/outline'
import {
  ChartBarIcon as ChartBarIconSolid,
  ShieldExclamationIcon as ShieldExclamationIconSolid,
  ChartPieIcon as ChartPieIconSolid,
} from '@heroicons/vue/24/solid'

const route = useRoute()
const isCollapsed = ref(false)

const navItems = [
  { path: '/', label: 'Dashboard', icon: ChartBarIcon, activeIcon: ChartBarIconSolid },
  { path: '/risk', label: 'Risk Analysis', icon: ShieldExclamationIcon, activeIcon: ShieldExclamationIconSolid },
  { path: '/forecast', label: 'Forecast', icon: ChartPieIcon, activeIcon: ChartPieIconSolid },
]

function isActive(path: string) {
  if (path === '/') return route.path === '/'
  return route.path.startsWith(path)
}

function toggleSidebar() {
  isCollapsed.value = !isCollapsed.value
}
</script>

<template>
  <div class="min-h-screen flex">
    <!-- Sidebar -->
    <aside
      :class="[
        'fixed top-0 left-0 h-screen bg-surface-900/50 backdrop-blur-xl border-r border-surface-800/50 flex flex-col z-40 transition-all duration-300',
        isCollapsed ? 'w-20' : 'w-72'
      ]"
    >
      <!-- Logo -->
      <div class="h-20 flex items-center px-4">
        <RouterLink to="/" class="flex items-center gap-3 group" :class="isCollapsed ? 'justify-center w-full' : ''">
          <div class="w-10 h-10 bg-gradient-accent rounded-xl flex items-center justify-center shadow-glow group-hover:shadow-glow transition-shadow flex-shrink-0">
            <ChartBarIconSolid class="w-5 h-5 text-surface-950" />
          </div>
          <div v-if="!isCollapsed" class="overflow-hidden">
            <span class="text-lg font-semibold text-surface-50 tracking-tight">Revenue</span>
            <span class="text-lg font-semibold text-gradient tracking-tight"> Intel</span>
          </div>
        </RouterLink>
      </div>

      <!-- Navigation -->
      <nav class="flex-1 px-3 py-6 space-y-2 overflow-hidden">
        <p v-if="!isCollapsed" class="px-3 mb-4 text-xs font-medium text-surface-500 uppercase tracking-wider">Navigation</p>
        <RouterLink
          v-for="item in navItems"
          :key="item.path"
          :to="item.path"
          :class="[
            'group flex items-center gap-3 px-4 py-3 rounded-xl text-sm font-medium transition-all duration-200',
            isCollapsed ? 'justify-center' : '',
            isActive(item.path)
              ? 'bg-accent-500/10 text-accent-400 shadow-glow-sm'
              : 'text-surface-400 hover:text-surface-200 hover:bg-surface-800/50'
          ]"
          :title="isCollapsed ? item.label : ''"
        >
          <component
            :is="isActive(item.path) ? item.activeIcon : item.icon"
            :class="[
              'w-5 h-5 transition-colors flex-shrink-0',
              isActive(item.path) ? 'text-accent-400' : 'text-surface-500 group-hover:text-surface-300'
            ]"
          />
          <span v-if="!isCollapsed" class="truncate">{{ item.label }}</span>
          <div
            v-if="isActive(item.path) && !isCollapsed"
            class="ml-auto w-2 h-2 rounded-full bg-accent-400 animate-pulse-slow flex-shrink-0"
          />
        </RouterLink>
      </nav>

      <!-- Collapse Toggle Button -->
      <button
        @click="toggleSidebar"
        class="absolute top-1/2 -right-3 transform -translate-y-1/2 w-6 h-6 bg-surface-800 border border-surface-700 rounded-full flex items-center justify-center text-surface-400 hover:text-surface-200 hover:bg-surface-700 transition-colors z-50"
      >
        <ChevronLeftIcon v-if="!isCollapsed" class="w-3.5 h-3.5" />
        <ChevronRightIcon v-else class="w-3.5 h-3.5" />
      </button>

      <!-- Footer -->
      <div v-if="!isCollapsed" class="px-4 py-6 border-t border-surface-800/50">
        <div class="px-4 py-3 rounded-xl bg-surface-800/30 border border-surface-700/30">
          <p class="text-xs text-surface-500 font-medium">Model Version</p>
          <p class="text-sm text-surface-300 font-mono mt-1">LightGBM v2.1</p>
          <div class="flex items-center gap-3 mt-2">
            <span class="text-xs text-surface-500">AUC: <span class="text-accent-400 font-mono">0.58</span></span>
            <span class="text-xs text-surface-500">ECE: <span class="text-accent-400 font-mono">0.031</span></span>
          </div>
        </div>
      </div>
    </aside>

    <!-- Main Content -->
    <main
      :class="[
        'flex-1 overflow-auto min-h-screen transition-all duration-300',
        isCollapsed ? 'ml-20' : 'ml-72'
      ]"
    >
      <div class="max-w-7xl mx-auto px-8 py-8">
        <RouterView />
      </div>
    </main>
  </div>
</template>
