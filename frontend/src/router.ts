import { createRouter, createWebHistory } from 'vue-router'
import DashboardView from './views/DashboardView.vue'
import RiskView from './views/RiskView.vue'
import DealView from './views/DealView.vue'
import ForecastView from './views/ForecastView.vue'

const routes = [
  {
    path: '/',
    name: 'dashboard',
    component: DashboardView,
  },
  {
    path: '/risk',
    name: 'risk',
    component: RiskView,
  },
  {
    path: '/deals/:id',
    name: 'deal',
    component: DealView,
  },
  {
    path: '/forecast',
    name: 'forecast',
    component: ForecastView,
  },
]

const router = createRouter({
  history: createWebHistory(),
  routes,
})

export default router
