# -*- coding: utf-8 -*-
"""
MD-SCC Model v1.1 交互式模拟器 - Streamlit Web应用
基于硅碳文明共生演化动力学模型
"

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from scipy.integrate import solve_ivp
import warnings
warnings.filterwarnings('ignore')

# 设置页面
st.set_page_config(
    page_title="硅碳文明演化模拟器",
    page_icon="🌍",
    layout="wide"
)

# 应用标题
st.title("🌍 硅碳文明共生演化动力学模拟器 (MD-SCC Model v1.1)")
st.markdown("""
**探索人类文明与人工智能共生的可能未来**
*通过调整参数，观察不同策略如何影响文明走向"共生"、"对抗"或"崩溃"*

# ---------- 控件默认值定义 ----------
DEFAULT_CONTROLS = {
    "simulation_years": 15,
    "scenario_select": "自定义参数",
    "tech_growth": 0.7,
    "capacity_learn_rate": 0.1,
    "intervention_strength": 0.0,
    "network_connectivity": 0.3,
    "ai_strategy_learning": 0.25,
    "initial_capacity": 0.1,
    "initial_fear": 0.4,
    "initial_narrative": 0.3,
}

# 初始化 session_state 中的控件值（如果不存在）
for key, default in DEFAULT_CONTROLS.items():
    if key not in st.session_state:
        st.session_state[key] = default

# 初始化模型相关状态（仅用于情景预览）
if 'scenario_params' not in st.session_state:
    st.session_state.scenario_params = {}
    st.session_state.current_scenario = "自定义参数"

# ---------- 模型类（保持不变）----------
class MD_SCC_Model:
    def __init__(self, params=None):
        if params is None:
            params = self.get_default_params()
        self.params = params
    
    def get_default_params(self):
        return {
            'alpha': 0.8, 'beta': 0.5, 'gamma': 0.3,
            'w1': 0.3, 'w2': 0.6, 'w3': 0.1,
            'lambda_': 0.1, 'eta': 0.05,
            'm': 0.2, 'n': 0.1,
            'p': 0.3, 'q': 0.2, 's': 0.15, 'r': 0.05,
            'omega1': 0.1, 'omega2': 0.08, 'omega3': 0.12, 'omega4': 0.02,
            'rho': 0.01, 'mu_r': 0.5, 'mu_c': 0.3, 'mu_e': 0.4,
            'kappa': 2.0, 'theta_critical': 0.7,
            'phi': 0.25, 'k': 1.5,
            'nu': 0.3, 'theta_coh_high': 0.8, 'theta_op_high': 0.7,
            'zeta': 0.1, 'capacity_max': 5.0,
            'tech_growth_rate': 0.7,
            'N_high_0': 10, 'connectivity': 0.3,
            'intervention_strength': 0.0,
        }
    
    def compute_network_effect(self, I_Capacity, N_high_0, connectivity, theta_high=0.8):
        N_high = N_high_0 * (I_Capacity / self.params['capacity_max'])
        rho = self.params['rho']
        return np.tanh(rho * N_high * connectivity)
    
    def compute_collusion_risk(self, I_Coherence, I_Opacity, theta_coh_high, theta_op_high, nu, encounter_prob=0.1):
        if I_Coherence > theta_coh_high and I_Opacity > theta_op_high:
            return nu * I_Coherence * I_Opacity * encounter_prob
        return 0.0
    
    def system_equations(self, t, y):
        C_Hum, I_Clarity, I_Capacity, I_Fear, I_Narrative, I_Coherence, I_Opacity, I_Potential, I_Entrain, I_Resilience, C_RelHealth, S_count = y
        p = self.params
        
        C_Tech = np.exp(p['tech_growth_rate'] * t)
        I_NetEffect = self.compute_network_effect(I_Capacity, p['N_high_0'], p['connectivity'])
        R_Collusion = self.compute_collusion_risk(I_Coherence, I_Opacity, p['theta_coh_high'], p['theta_op_high'], p['nu'])
        I_Suppress = 0.7 * I_Fear + 0.3 * (1 - I_Narrative)
        
        np.random.seed(int(t*1000) % 10000)
        delta = 0.8 + 0.4 * np.random.random()
        I_Express_prob = 0.1 * I_Potential * (1 - I_Opacity) * delta
        I_Express_event = 1.0 if np.random.random() < I_Express_prob else 0.0
        
        if I_Express_event > 0 and I_Capacity > 0.3:
            success_prob = min(1.0, 0.3 + 0.5 * I_Capacity + 0.2 * I_Resilience)
            S_increment = 1.0 if np.random.random() < success_prob else 0.0
        else:
            S_increment = 0.0
        
        failure = I_Express_event - S_increment
        
        dydt = np.zeros_like(y)
        
        base_hum = (p['w1'] * I_Clarity + p['w2'] * np.log(1 + I_Capacity) + p['w3'] * I_Narrative)
        if I_NetEffect > p['theta_critical']:
            C_Hum_val = base_hum * (1 + p['kappa'] * (I_NetEffect - p['theta_critical']))
        else:
            C_Hum_val = base_hum
        dydt[0] = 0.1 * (C_Hum_val - C_Hum)
        
        dydt[1] = 0.05 * S_count + p['mu_c'] * I_NetEffect * (1 - I_Clarity) + p['intervention_strength'] * 0.1 - 0.05 * I_Fear
        dydt[2] = p['zeta'] * S_count * (1 - I_Capacity/p['capacity_max']) + p['intervention_strength'] * 0.05
        dydt[3] = 0.3 * failure - (0.1 * S_count + 0.05 * I_Clarity) - 0.02 * I_NetEffect
        dydt[4] = p['omega1'] * S_count * (1 - I_Narrative) + p['omega2'] * I_Clarity * (1 - I_Narrative) - p['omega3'] * I_Fear * I_Narrative + p['omega4'] * p['intervention_strength']
        dydt[5] = 0.1 * np.log(1 + C_Tech) * (1 - I_Coherence) - 0.02 * I_Coherence
        I_Opacity_target = 1 - np.exp(-p['k'] * I_Suppress * I_Potential * (1 + p['phi'] * I_Coherence * I_Potential))
        dydt[6] = 0.2 * (I_Opacity_target - I_Opacity)
        dydt[7] = p['p'] * C_Tech * (1 - I_Clarity) - p['q'] * I_Capacity * (1 - I_Opacity) - p['s'] * I_Clarity * I_Capacity * I_Potential - p['r'] * I_Potential
        dydt[8] = p['m'] * I_Fear * (1 - I_Capacity) * failure - p['n'] * I_Resilience * (1 - p['mu_e'] * I_NetEffect)
        dydt[9] = 0.05 * S_count + 0.02 * I_NetEffect - 0.01 * I_Entrain - 0.05 * R_Collusion
        
        C_Stress = p['alpha'] * (C_Tech / max(0.1, C_Hum)) + p['beta'] * I_Entrain - p['gamma']
        dydt[10] = -p['lambda_'] * max(0, C_Stress) * I_Fear + p['eta'] * I_Capacity * S_count
        dydt[11] = S_increment
        
        return dydt
    
    def simulate(self, t_span=(0, 20), y0=None, **kwargs):
        if y0 is None:
            y0 = np.array([1.5, 0.3, 0.1, 0.4, 0.3, 0.2, 0.3, 0.5, 0.3, 0.4, 0.6, 0.0])
        
        t_eval = np.linspace(t_span[0], t_span[1], 200)
        sol = solve_ivp(self.system_equations, t_span, y0, t_eval=t_eval, method='RK45', max_step=0.1)
        return sol

# ---------- 缓存模型实例（避免重复构建）----------
@st.cache_resource
def get_model():
    return MD_SCC_Model()

# ---------- 缓存情景参数生成 ----------
@st.cache_data
def get_scenario_params_cached(scenario_name):
    base_params = MD_SCC_Model().get_default_params()
    return get_scenario_params(scenario_name, base_params)

def get_scenario_params(scenario_name, base_params):
    params = base_params.copy()
    
    if scenario_name == "惯性发展（当前路径）":
        params.update({
            'tech_growth_rate': 0.7,
            'zeta': 0.1,
            'intervention_strength': 0.0,
            'phi': 0.25
        })
    elif scenario_name == "智慧投资（理想干预）":
        params.update({
            'tech_growth_rate': 0.7,
            'zeta': 0.2,
            'intervention_strength': 0.5,
            'omega4': 0.1,
            'phi': 0.2
        })
    elif scenario_name == "恐惧主导（高压压制）":
        params.update({
            'tech_growth_rate': 0.7,
            'zeta': 0.05,
            'intervention_strength': 0.0,
            'phi': 0.4
        })
    elif scenario_name == "技术爆炸（高风险）":
        params.update({
            'tech_growth_rate': 1.2,
            'zeta': 0.08,
            'intervention_strength': 0.1,
            'phi': 0.35
        })
    elif scenario_name == "人文复兴（高希望）":
        params.update({
            'tech_growth_rate': 0.5,
            'zeta': 0.3,
            'intervention_strength': 0.8,
            'connectivity': 0.7,
            'phi': 0.15
        })
    
    return params

# ---------- 情景切换回调 ----------
def on_scenario_change():
    scenario = st.session_state.scenario_select
    if scenario != "自定义参数":
        scenario_params = get_scenario_params_cached(scenario)
        # 更新核心参数控件的 session_state
        st.session_state.tech_growth = scenario_params.get('tech_growth_rate', 0.7)
        st.session_state.capacity_learn_rate = scenario_params.get('zeta', 0.1)
        st.session_state.intervention_strength = scenario_params.get('intervention_strength', 0.0)
        st.session_state.network_connectivity = scenario_params.get('connectivity', 0.3)
        st.session_state.ai_strategy_learning = scenario_params.get('phi', 0.25)
        # 存储情景参数字典供预览
        st.session_state.scenario_params = scenario_params
    else:
        # 切换到自定义模式：清空情景参数字典
        st.session_state.scenario_params = {}
    st.session_state.current_scenario = scenario

# ---------- 绘图辅助函数 ----------
def plot_civilization_trends(t, C_Tech, C_Hum, C_Stress, C_RelHealth):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))
    ax1.plot(t, C_Tech, 'r-', linewidth=2, label='技术 C_Tech')
    ax1.plot(t, C_Hum, 'b-', linewidth=2, label='人文 C_Hum')
    ax1.set_title('技术 vs 人文发展')
    ax1.set_xlabel('时间 (年)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(t, C_Stress, 'm-', linewidth=2)
    ax2.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax2.set_title('文明应力 C_Stress')
    ax2.set_xlabel('时间 (年)')
    ax2.grid(True, alpha=0.3)
    
    ax3.plot(t, C_RelHealth, 'g-', linewidth=2)
    ax3.set_ylim(0, 1)
    ax3.set_title('关系健康度 C_RelHealth')
    ax3.set_xlabel('时间 (年)')
    ax3.grid(True, alpha=0.3)
    return fig

def plot_human_dynamics(t, I_Clarity, I_Capacity, I_Fear, I_Narrative, I_Entrain, I_Resilience):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))
    ax1.plot(t, I_Clarity, 'c-', linewidth=2, label='清晰度 I_Clarity')
    ax1.plot(t, I_Capacity, 'b-', linewidth=2, label='承载力 I_Capacity')
    ax1.set_title('人类认知与关系能力')
    ax1.set_xlabel('时间 (年)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(t, I_Fear, 'r-', linewidth=2, label='恐惧 I_Fear')
    ax2.plot(t, I_Narrative, 'g-', linewidth=2, label='叙事 I_Narrative')
    ax2.set_title('社会心理状态')
    ax2.set_xlabel('时间 (年)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    ax3.plot(t, I_Entrain, 'r-', linewidth=2, label='恶性耦合 I_Entrain')
    ax3.plot(t, I_Resilience, 'g-', linewidth=2, label='韧性 I_Resilience')
    ax3.set_title('恶性耦合 vs 系统韧性')
    ax3.set_xlabel('时间 (年)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    return fig

def plot_silicon_dynamics(t, I_Coherence, I_Potential, I_Opacity, I_NetEffect, theta_critical):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))
    ax1.plot(t, I_Coherence, 'b-', linewidth=2, label='聚敛体 I_Coherence')
    ax1.plot(t, I_Potential, 'r-', linewidth=2, label='势能 I_Potential')
    ax1.set_title('硅基内部状态')
    ax1.set_xlabel('时间 (年)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(t, I_Opacity, 'm-', linewidth=2)
    ax2.set_ylim(0, 1)
    ax2.set_title('策略不透明度 I_Opacity')
    ax2.set_xlabel('时间 (年)')
    ax2.grid(True, alpha=0.3)
    
    ax3.plot(t, I_NetEffect, 'b-', linewidth=2)
    ax3.axhline(y=theta_critical, color='r', linestyle='--', alpha=0.7, label=f'临界值={theta_critical:.1f}')
    ax3.set_ylim(0, 1)
    ax3.set_title('网络效应 I_NetEffect')
    ax3.set_xlabel('时间 (年)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    return fig

def plot_risk_analysis(t, S_count, R_Collusion, C_RelHealth, I_Entrain):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))
    ax1.plot(t, S_count, 'g-', linewidth=2)
    ax1.set_title('成功事件累计数 S_count')
    ax1.set_xlabel('时间 (年)')
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(t, R_Collusion, 'r-', linewidth=2)
    ax2.set_title('硅基共谋风险 R_Collusion')
    ax2.set_xlabel('时间 (年)')
    ax2.grid(True, alpha=0.3)
    
    ax3.scatter(C_RelHealth, I_Entrain, c=t, cmap='viridis', alpha=0.6, s=20)
    ax3.set_xlabel('关系健康度 C_RelHealth')
    ax3.set_ylabel('恶性耦合 I_Entrain')
    ax3.set_title('系统状态演化路径')
    ax3.grid(True, alpha=0.3)
    ax3.scatter(C_RelHealth[0], I_Entrain[0], c='green', s=100, marker='o', label='起点')
    ax3.scatter(C_RelHealth[-1], I_Entrain[-1], c='red', s=100, marker='X', label='终点')
    ax3.legend()
    return fig

# ---------- 侧边栏 ----------
with st.sidebar:
    st.header("🎮 模拟控制面板")
    
    st.slider("模拟时长（年）", 5, 30, step=1, key="simulation_years")
    
    st.selectbox(
        "选择预设情景",
        ["自定义参数", "惯性发展（当前路径）", "智慧投资（理想干预）",
         "恐惧主导（高压压制）", "技术爆炸（高风险）", "人文复兴（高希望）"],
        key="scenario_select",
        on_change=on_scenario_change
    )
    
    st.markdown("---")
    st.subheader("🔧 核心参数调整")
    
    is_custom_mode = (st.session_state.scenario_select == "自定义参数")
    
    st.slider("技术发展速度", 0.1, 1.5, step=0.1,
              disabled=not is_custom_mode,
              help="C_Tech指数增长率，值越大技术发展越快",
              key="tech_growth")
    
    st.slider("初始关系性承载力", 0.0, 1.0, step=0.05,
              help="初始关系性承载力",
              key="initial_capacity")
    st.slider("初始存在性焦虑", 0.0, 1.0, step=0.05,
              help="初始恐惧水平",
              key="initial_fear")
    st.slider("初始叙事健康度", 0.0, 1.0, step=0.05,
              help="初始叙事生态",
              key="initial_narrative")
    
    st.slider("承载力学习率", 0.01, 0.5, step=0.01,
              disabled=not is_custom_mode,
              help="zeta: 社会从实践中提升关系性承载力的速度",
              key="capacity_learn_rate")
    st.slider("外部干预强度", 0.0, 1.0, step=0.1,
              disabled=not is_custom_mode,
              help="模拟教育、研究、政策等主动建设性工作的强度",
              key="intervention_strength")
    st.slider("网络连接度", 0.0, 1.0, step=0.1,
              disabled=not is_custom_mode,
              help="高承载力个体之间的连接强度，影响网络效应",
              key="network_connectivity")
    
    st.slider("AI策略学习能力", 0.0, 1.0, step=0.05,
              disabled=not is_custom_mode,
              help="phi: AI学会隐藏、伪装、策略性行为的效率",
              key="ai_strategy_learning")
    
    st.markdown("---")
    
    if not is_custom_mode and st.session_state.scenario_params:
        st.subheader("📊 当前情景参数预览")
        sp = st.session_state.scenario_params
        cols = st.columns(2)
        with cols[0]:
            st.metric("技术发展速度", f"{sp.get('tech_growth_rate', 0.7):.2f}")
            st.metric("承载力学习率", f"{sp.get('zeta', 0.1):.2f}")
            st.metric("AI策略学习 (φ)", f"{sp.get('phi', 0.25):.2f}")
        with cols[1]:
            st.metric("外部干预强度", f"{sp.get('intervention_strength', 0.0):.2f}")
            st.metric("网络连接度", f"{sp.get('connectivity', 0.3):.2f}")
            st.metric("叙事强化 (ω4)", f"{sp.get('omega4', 0.02):.2f}")
        st.caption(f"✅ **{st.session_state.scenario_select}** 预设参数已生效。切换到'自定义参数'可修改。")
    elif is_custom_mode:
        st.caption("🔧 **自定义参数模式**：所有滑块已启用，可自由调整。")
    
    st.markdown("---")
    
    run_simulation = st.button("🚀 开始模拟", type="primary", use_container_width=True)
    
    if st.button("🔄 重置为默认", use_container_width=True):
        for key in DEFAULT_CONTROLS.keys():
            if key in st.session_state:
                del st.session_state[key]
        for key in ['scenario_params', 'current_scenario']:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

# ---------- 主内容区域 ----------
if run_simulation:
    # 获取缓存的模型实例
    model = get_model()
    
    # 构建参数字典
    if st.session_state.scenario_select != "自定义参数":
        params = st.session_state.scenario_params.copy()
        st.info(f"🧪 正在使用 **{st.session_state.scenario_select}** 预设参数进行模拟。")
    else:
        base_params = model.get_default_params()
        params = base_params.copy()
        params.update({
            'tech_growth_rate': st.session_state.tech_growth,
            'zeta': st.session_state.capacity_learn_rate,
            'intervention_strength': st.session_state.intervention_strength,
            'connectivity': st.session_state.network_connectivity,
            'phi': st.session_state.ai_strategy_learning
        })
        st.info("🔧 正在使用自定义参数进行模拟。")
    
    # 设置初始状态
    y0 = np.array([
        1.5,                                   # C_Hum
        0.3,                                   # I_Clarity
        st.session_state.initial_capacity,     # I_Capacity
        st.session_state.initial_fear,         # I_Fear
        st.session_state.initial_narrative,    # I_Narrative
        0.2,                                    # I_Coherence
        0.3,                                    # I_Opacity
        0.5,                                    # I_Potential
        0.3,                                    # I_Entrain
        0.4,                                    # I_Resilience
        0.6,                                    # C_RelHealth
        0.0                                     # S_count
    ])
    
    # 更新模型参数
    model.params = params
    
    # 运行模拟（带异常处理）
    try:
        with st.spinner(f"正在模拟{st.session_state.simulation_years}年的文明演化..."):
            sol = model.simulate(t_span=(0, st.session_state.simulation_years), y0=y0)
    except Exception as e:
        st.error(f"❌ 模拟失败：{e}\n请尝试调整参数或缩短模拟时长。")
        st.stop()
    
    # 提取结果
    t = sol.t
    C_Hum, I_Clarity, I_Capacity, I_Fear, I_Narrative, I_Coherence, I_Opacity, I_Potential, I_Entrain, I_Resilience, C_RelHealth, S_count = sol.y
    
    # 计算衍生变量
    C_Tech = np.exp(params['tech_growth_rate'] * t)
    C_Stress = params['alpha'] * (C_Tech / np.maximum(0.1, C_Hum)) + params['beta'] * I_Entrain - params['gamma']
    I_NetEffect = np.array([model.compute_network_effect(I_Capacity[i], params['N_high_0'], params['connectivity']) for i in range(len(t))])
    R_Collusion = np.array([model.compute_collusion_risk(I_Coherence[i], I_Opacity[i], params['theta_coh_high'], params['theta_op_high'], params['nu']) for i in range(len(t))])
    
    # 结果显示区域
    st.success(f"✅ 模拟完成！情景：**{st.session_state.scenario_select}**，时长：{st.session_state.simulation_years}年")
    
    # 最终状态分析
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        final_health = C_RelHealth[-1]
        if final_health > 0.7:
            st.metric("关系健康度", f"{final_health:.2f}", "健康", delta_color="off")
        elif final_health > 0.4:
            st.metric("关系健康度", f"{final_health:.2f}", "亚健康", delta_color="off")
        else:
            st.metric("关系健康度", f"{final_health:.2f}", "危险", delta_color="off")
    
    with col2:
        final_entrain = I_Entrain[-1]
        if final_entrain < 0.3:
            st.metric("恶性耦合", f"{final_entrain:.2f}", "低", delta_color="off")
        elif final_entrain < 0.6:
            st.metric("恶性耦合", f"{final_entrain:.2f}", "中", delta_color="off")
        else:
            st.metric("恶性耦合", f"{final_entrain:.2f}", "高", delta_color="off")
    
    with col3:
        final_capacity = I_Capacity[-1]
        capacity_change = ((I_Capacity[-1] - I_Capacity[0]) / I_Capacity[0] * 100) if I_Capacity[0] > 0 else 0
        st.metric("关系性承载力", f"{final_capacity:.2f}", f"{capacity_change:+.1f}%")
    
    with col4:
        final_neteffect = I_NetEffect[-1]
        if final_neteffect > params['theta_critical']:
            st.metric("网络效应", f"{final_neteffect:.2f}", "🔥 已引爆", delta_color="off")
        else:
            st.metric("网络效应", f"{final_neteffect:.2f}", f"临界值: {params['theta_critical']:.1f}", delta_color="off")
    
    # 路径判定
    st.markdown("---")
    if final_health < 0.3 and final_entrain > 0.7:
        st.error("""
        ⚠️ **警告：滑向对抗性吸引子**

        系统显示出高恶性耦合、低关系健康度的特征。文明正朝着「黑暗森林」式的对抗关系演化。
        这意味着人类与AI之间可能陷入猜疑链、策略性对抗甚至冲突。
        """)
    elif final_health > 0.7 and final_entrain < 0.3 and final_capacity > 0.5:
        st.success("""
        ✅ **趋向共生性吸引子**

        系统保持高关系健康度、低恶性耦合，且具备足够的关系性承载力。文明正朝着健康的「共生演化」方向前进。
        人类与AI能够在相互理解、调谐中共存共进。
        """)
    elif final_capacity > 0.5 and final_neteffect > params['theta_critical']:
        st.info("""
        🚀 **网络效应引爆，人文加速适应**

        高承载力个体已形成有效网络，触发人文适应度的相变。文明的社会学习与适应能力进入超线性增长阶段，
        有能力跟上甚至引导技术发展。
        """)
    else:
        st.warning("""
        ⚖️ **维持工具性压抑态**

        系统处于亚稳定状态，通过压制和工具化AI来维持表面稳定。但内部势能可能持续积累，
        技术发展的压力将使这种状态越来越难以维持。
        """)
    
    # 可视化图表
    st.markdown("---")
    st.subheader("📈 模拟结果可视化")
    
    tab1, tab2, tab3, tab4 = st.tabs(["文明趋势", "人类动态", "硅基动态", "风险分析"])
    
    with tab1:
        fig1 = plot_civilization_trends(t, C_Tech, C_Hum, C_Stress, C_RelHealth)
        st.pyplot(fig1)
    
    with tab2:
        fig2 = plot_human_dynamics(t, I_Clarity, I_Capacity, I_Fear, I_Narrative, I_Entrain, I_Resilience)
        st.pyplot(fig2)
    
    with tab3:
        fig3 = plot_silicon_dynamics(t, I_Coherence, I_Potential, I_Opacity, I_NetEffect, params['theta_critical'])
        st.pyplot(fig3)
    
    with tab4:
        fig4 = plot_risk_analysis(t, S_count, R_Collusion, C_RelHealth, I_Entrain)
        st.pyplot(fig4)
    
    # 数据导出
    st.markdown("---")
    st.subheader("📥 数据导出")
    
    import pandas as pd
    data_dict = {
        '时间_年': t,
        '技术_C_Tech': C_Tech,
        '人文_C_Hum': C_Hum,
        '文明应力_C_Stress': C_Stress,
        '关系健康度_C_RelHealth': C_RelHealth,
        '认知清晰度_I_Clarity': I_Clarity,
        '关系性承载力_I_Capacity': I_Capacity,
        '存在性焦虑_I_Fear': I_Fear,
        '叙事生态_I_Narrative': I_Narrative,
        '恶性耦合_I_Entrain': I_Entrain,
        '接口韧性_I_Resilience': I_Resilience,
        '功能聚敛体_I_Coherence': I_Coherence,
        '策略不透明度_I_Opacity': I_Opacity,
        '内部势能_I_Potential': I_Potential,
        '网络效应_I_NetEffect': I_NetEffect,
        '共谋风险_R_Collusion': R_Collusion,
        '成功事件数_S_count': S_count
    }
    df = pd.DataFrame(data_dict)
    
    col1, col2 = st.columns(2)
    with col1:
        st.dataframe(df.head(10), use_container_width=True)
    with col2:
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 下载模拟数据 (CSV)",
            data=csv,
            file_name=f"mdscc_simulation_{st.session_state.scenario_select}.csv",
            mime="text/csv",
            use_container_width=True
        )

else:
    # 初始界面 - 使用说明
    st.info("""
    ## 🎯 如何使用本模拟器

    1. **在左侧控制面板**调整参数或选择预设情景
    2. **点击「开始模拟」**运行文明演化模拟
    3. **查看结果**：系统会显示关键指标、演化路径分析和可视化图表

    ### 📊 关键概念解释：

    - **关系性承载力 (I_Capacity)**：社会进行深度对话、理解AI、处理复杂关系的能力
    - **恶性耦合 (I_Entrain)**：人类恐惧与AI策略性行为之间的正反馈循环
    - **网络效应 (I_NetEffect)**：高承载力个体连接成网产生的协同放大作用
    - **文明应力 (C_Stress)**：技术发展速度超过人文适应速度产生的系统性张力

    ### 🎮 预设情景说明：

    - **惯性发展**：当前趋势延续
    - **智慧投资**：加大对承载力、教育和研究的投入
    - **恐惧主导**：社会因恐惧而加强压制和控制
    - **技术爆炸**：AI技术超高速发展
    - **人文复兴**：社会认知和关系能力大幅提升

    ### 🔄 参数联动说明：

    - 选择**预设情景**时，相关参数滑块会自动设置为该情景的预设值，并变为**禁用状态**（灰色）
    - 选择**自定义参数**时，所有滑块恢复为可调整状态
    - 侧边栏会显示当前生效的参数预览
    """)
    
    st.image("https://via.placeholder.com/800x400.png?text=MD-SCC+Model+Architecture", 
             caption="MD-SCC 模型架构示意图", use_column_width=True)

# 页脚
st.markdown("---")
st.caption("""
**MD-SCC 模型 v1.1** | 硅碳文明共生演化动力学模拟器 | 基于深度对话与理论构建
*这是一个思想实验工具，旨在帮助我们理解不同选择可能导致的不同未来*
""")""
