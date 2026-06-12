const defaultGoal = "研究机械臂在不同物体、接触条件和任务目标下的操作成功率、失败类型和下一轮实验改进方向";

let experimentSpace = {
  skill_id: ["pick_lift", "reach_touch", "contact_sweep", "tool_contact_sweep", "peg_insert"],
  object_id: ["rect_block", "cube_5cm", "cylinder_can", "cube_7cm", "screw_head", "insertion_socket"],
  tool_id: ["hammer", "spatula", "screwdriver", "peg"],
  object_position: ["center", "left", "right"],
  friction: ["low", "medium", "high"],
  grasp_height_delta: ["nominal", "low", "high"],
  sweep_scale: ["short", "nominal", "long"]
};

let currentTask = {
  task_id: "fr3_arm_primitives",
  title: "General FR3 arm primitive experiments",
  description: "默认任务：用真实 FR3 + Franka Hand MuJoCo 场景运行触碰、抓取抬升和接触扫动基础实验。",
  experiment_space: experimentSpace,
  execution_kind: "robot_arm_skill_simulation",
  manipulation_actor: "Franka FR3 arm with Franka Hand"
};

let hypotheses = [
  {
    id: "H1",
    title: "位姿扰动鲁棒性",
    text: "物体初始位置扰动越大，抓取成功率越低。",
    metric: "success_rate"
  },
  {
    id: "H2",
    title: "接触参数影响",
    text: "低摩擦条件下，失败更可能表现为滑落。",
    metric: "failure_type"
  },
  {
    id: "H3",
    title: "抓取高度偏差",
    text: "抓取高度偏差会改变抓空、滑落和最终距离。",
    metric: "final_distance"
  }
];

let currentRuns = [];
let currentSummary = null;
let currentReport = "";
let currentTrace = null;
let currentAgentDesign = null;
let currentAgentAnalysis = null;
let currentAgentTrace = [];
let currentAgentApiCalls = null;
let currentExecutionPlan = null;
let currentAgentProvider = null;
let objectLibrary = [
  { object_id: "cube_5cm", name_zh: "5cm 立方体", name_en: "5 cm cube", tags: ["graspable", "pushable"] },
  { object_id: "rect_block", name_zh: "长方体积木", name_en: "rectangular block", tags: ["graspable", "pushable"] },
  { object_id: "cylinder_can", name_zh: "圆柱罐", name_en: "cylindrical can", tags: ["graspable", "pushable"] },
  { object_id: "small_sphere", name_zh: "小球", name_en: "small sphere", tags: ["pushable"] },
  { object_id: "flat_puck", name_zh: "扁圆盘", name_en: "flat puck", tags: ["pushable"] }
];
let assetRegistry = { robot_assets: [], task_blueprints: [] };
let taskCatalog = [];
let replayHandle = null;
let worldBounds = null;
let runTimer = null;
let robotCatalog = [];
let currentNlpResult = null;

const $ = (id) => document.getElementById(id);
const pct = (value) => `${(Number(value || 0) * 100).toFixed(1)}%`;

function outputLanguage() {
  return $("outputLanguage")?.value || "zh";
}

function isEnglish() {
  return outputLanguage() === "en";
}

function objectName(obj) {
  return isEnglish() ? (obj.name_en || obj.object_id) : (obj.name_zh || obj.object_id);
}

function formatInlineList(values, emptyLabel) {
  const items = (values || []).filter(Boolean);
  if (!items.length) return emptyLabel;
  return items.slice(0, 5).join(isEnglish() ? ", " : "、");
}

function taskTypeLabel(type) {
  const labels = {
    pick_and_place: isEnglish() ? "Pick and place" : "抓取放置",
    push: isEnglish() ? "Pushing" : "推动",
    press: isEnglish() ? "Button press" : "按压按钮",
    touch: isEnglish() ? "Contact / touch" : "接触",
    insert: isEnglish() ? "Insertion" : "插入装配",
    screwdriving: isEnglish() ? "Screwdriving" : "拧螺丝",
    tool_use: isEnglish() ? "Tool use" : "工具操作",
    general: isEnglish() ? "General manipulation" : "通用操作",
  };
  return labels[type] || type || (isEnglish() ? "Generated task" : "生成任务");
}

function compactTaskDescription(taskSpec, payload = currentNlpResult) {
  if (!taskSpec) return "";
  const robot = payload?.robot_spec?.name || payload?.robot_id || taskSpec.robot_preference || "auto";
  const endEffector = payload?.robot_spec?.end_effector_name || (payload?.robot_spec?.has_gripper ? "parallel gripper" : "none");
  const objects = (taskSpec.objects || []).map((obj) => obj.object_id || obj);
  const variables = Object.keys(taskSpec.experiment_variables || {});
  const actionCount = (taskSpec.actions || []).length;
  const none = isEnglish() ? "none" : "无";
  const lines = isEnglish()
    ? [
        `Task: ${taskTypeLabel(taskSpec.task_type)}`,
        `Robot: ${robot}`,
        `End effector: ${endEffector}`,
        `Scene: table workspace; objects ${formatInlineList(objects, none)}; held tool ${taskSpec.held_tool_id || none}`,
        `Plan: ${actionCount} motion primitives; variables ${formatInlineList(variables, none)}`,
      ]
    : [
        `任务：${taskTypeLabel(taskSpec.task_type)}`,
        `机器人：${robot}`,
        `末端执行器：${endEffector}`,
        `场景：桌面工作区；对象 ${formatInlineList(objects, none)}；手持工具 ${taskSpec.held_tool_id || none}`,
        `实验：${actionCount} 个运动原语；变量 ${formatInlineList(variables, none)}`,
      ];
  return lines.join("\n");
}

function buildNlpHypotheses(taskSpec) {
  const vars = Object.keys(taskSpec?.experiment_variables || {});
  const type = taskSpec?.task_type || "general";
  const primaryMetric = (taskSpec?.metrics || [])[0] || "success_rate";
  const title = taskTypeLabel(type);
  const rows = [];
  rows.push({
    id: "H1",
    title: isEnglish() ? `${title} success boundary` : `${title}成功边界`,
    text: isEnglish()
      ? "Vary the main task parameters and measure where success starts to degrade."
      : "扫描主要任务变量，观察成功率从稳定到下降的边界。",
    metric: primaryMetric,
  });
  if (vars.includes("object_position")) {
    rows.push({
      id: "H2",
      title: isEnglish() ? "Workspace sensitivity" : "工作空间敏感性",
      text: isEnglish()
        ? "Object placement changes reachability, contact quality, and failure mode."
        : "物体位置会改变可达性、接触质量和失败类型。",
      metric: "success_rate",
    });
  }
  if (vars.includes("press_depth") || vars.includes("insertion_depth") || vars.includes("push_force")) {
    rows.push({
      id: `H${rows.length + 1}`,
      title: isEnglish() ? "Interaction strength" : "交互强度",
      text: isEnglish()
        ? "Depth or force controls whether contact is effective or excessive."
        : "深度或力度决定接触是否有效，以及是否出现过度作用。",
      metric: type === "push" || type === "tool_use" ? "object_displacement" : "contact_steps",
    });
  }
  return rows.slice(0, 3);
}

function selectedTaskId() {
  const value = $("taskSelect")?.value || "";
  return value && value !== "__other__" ? value : "";
}

function effectiveGoal() {
  const base = $("goalInput").value.trim();
  const custom = $("customTaskInput")?.value.trim() || "";
  if (($("taskSelect")?.value || "") === "__other__" && custom) {
    return `${base}\n\n自定义任务：${custom}`;
  }
  return base;
}

const metricKeys = new Set([
  "run_id",
  "success",
  "failure_type",
  "trajectory_error",
  "collision_count",
  "final_distance",
  "max_grip_force",
  "cube_slip_detected",
  "touch_contact_duration",
  "contact_steps",
  "max_push_force",
  "overshoot",
  "lateral_error"
]);

function sleepFrame() {
  return new Promise((resolve) => requestAnimationFrame(() => resolve()));
}

function startRunFeedback(mode, limit) {
  const startTime = Date.now();
  const runBtn = $("runBtn");
  runBtn.disabled = true;
  runBtn.textContent = "运行中...";
  $("reportPreview").textContent = mode === "agent"
    ? "智能体正在运行：任务路由 -> 实验设计 -> MuJoCo 批量仿真 -> 结果分析。\n\n这一步会等待完整闭环结束后一次性返回结果，27 组通常也可能需要几十秒。"
    : "MuJoCo 正在运行本地批量仿真，请稍候。";

  const base = mode === "agent"
    ? `智能体闭环运行中 · ${limit} 组`
    : `MuJoCo 实验运行中 · ${limit} 组`;
  const update = () => {
    const seconds = Math.floor((Date.now() - startTime) / 1000);
    $("runStatus").textContent = `${base} · 已等待 ${seconds}s`;
  };
  update();
  runTimer = setInterval(update, 1000);
}

function stopRunFeedback() {
  if (runTimer) {
    clearInterval(runTimer);
    runTimer = null;
  }
  const runBtn = $("runBtn");
  runBtn.disabled = false;
  runBtn.textContent = "运行仿真实验";
}

function showToast(message, durationMs = 4200) {
  let container = $("toast-container");
  if (!container) {
    container = document.createElement("div");
    container.id = "toast-container";
    container.style.cssText = "position:fixed;top:16px;right:16px;z-index:9999;display:flex;flex-direction:column;gap:8px;";
    document.body.appendChild(container);
  }
  const toast = document.createElement("div");
  toast.style.cssText = "background:#1d2733;color:#fff;padding:12px 18px;border-radius:8px;font-size:14px;line-height:1.5;box-shadow:0 4px 16px rgba(0,0,0,0.18);max-width:420px;opacity:0;transition:opacity 0.2s;";
  toast.textContent = message;
  container.appendChild(toast);
  requestAnimationFrame(() => { toast.style.opacity = "1"; });
  setTimeout(() => {
    toast.style.opacity = "0";
    setTimeout(() => toast.remove(), 260);
  }, durationMs);
}

function taskVariables() {
  return Object.keys(currentTask?.experiment_space || experimentSpace || {});
}

function taskScopeLabel() {
  const taskId = currentTask?.task_id || "fr3_arm_primitives";
  if (taskId === "fr3_arm_primitives") {
    return "当前任务：通用 FR3 机械臂基础实验。智能体会先把目标落到已实现的触碰、抓取抬升、接触扫动技能，并扫描相关物体和接触变量。";
  }
  if (taskId === "tabletop_push") {
    return "当前任务：桌面推动。智能体已把输入目标映射到推动物块到目标区域这一类实验。";
  }
  if (taskId === "fr3_pick_place") {
    return "当前任务：FR3 抓取放置。智能体已把输入目标映射到夹爪抓取并放置物块这一类实验。";
  }
  return "当前任务来自后端注册表；未命中的新问题会先进入通用机械臂基础 runner，再根据目标扩展专用场景和技能。";
}

function taskExecutionLabel(task = currentTask) {
  const kind = task?.execution_kind || "";
  const actor = task?.manipulation_actor || "";
  if (kind.startsWith("robot_arm") || actor.toLowerCase().includes("arm")) {
    return isEnglish() ? "Robot-arm runner" : "机械臂 runner";
  }
  if (kind.includes("deformable")) {
    return isEnglish() ? "Deformable proxy runner" : "软体代理 runner";
  }
  return isEnglish() ? "Task-specific proxy runner" : "任务代理 runner";
}

function compactObjectLine(objects) {
  if (!objects.length) return isEnglish() ? "not loaded" : "未加载";
  const visible = objects.slice(0, 4).map(objectName);
  const more = objects.length > visible.length
    ? (isEnglish() ? ` +${objects.length - visible.length} more` : ` 等 ${objects.length} 个`)
    : "";
  return `${visible.join(isEnglish() ? ", " : "、")}${more}`;
}

function currentTaskUsesRobotArm() {
  return String(currentTask?.execution_kind || "").startsWith("robot_arm");
}

function inferVariableKeys(rows) {
  const fromTask = taskVariables().filter((key) => rows.some((row) => key in row));
  if (fromTask.length) return fromTask;
  const first = rows[0] || {};
  return Object.keys(first).filter((key) => !metricKeys.has(key));
}

function groupBy(rows, key) {
  return rows.reduce((acc, row) => {
    const value = String(row[key] ?? "unknown");
    acc[value] ||= [];
    acc[value].push(row);
    return acc;
  }, {});
}

async function fetchExperimentSpace(endpoint) {
  try {
    const resp = await fetch(endpoint.replace(/\/api\/run_experiments$/, "/api/experiment_space"));
    if (!resp.ok) return;
    const data = await resp.json();
    if (data.variables && typeof data.variables === "object") {
      experimentSpace = data.variables;
      currentTask.experiment_space = experimentSpace;
    }
  } catch {
    // Static mode remains usable when the API is offline.
  }
}

async function fetchObjectLibrary(endpoint) {
  try {
    const resp = await fetch(endpoint.replace(/\/api\/run_experiments$/, "/api/object_library"));
    if (!resp.ok) return;
    const data = await resp.json();
    if (Array.isArray(data.objects)) {
      objectLibrary = data.objects;
    }
  } catch {
    // Keep the built-in object list when the API is offline.
  }
}

async function fetchTaskCatalog(endpoint) {
  try {
    const resp = await fetch(endpoint.replace(/\/api\/run_experiments$/, "/api/tasks"));
    if (!resp.ok) return;
    const data = await resp.json();
    if (Array.isArray(data.tasks)) {
      taskCatalog = data.tasks;
      renderTaskSelect();
    }
  } catch {
    renderTaskSelect();
  }
}

function renderTaskSelect() {
  const select = $("taskSelect");
  if (!select) return;
  const current = select.value;
  const labels = {
    fr3_pick_place: isEnglish() ? "Franka FR3 pick-and-place" : "FR3 抓取放置",
    tabletop_push: isEnglish() ? "Tabletop object pushing" : "桌面推动",
    screwdriving: isEnglish() ? "Screwdriving" : "拧螺丝",
    tool_use: isEnglish() ? "Tool use" : "工具使用",
    assembly_insertion: isEnglish() ? "Assembly insertion" : "装配/插入",
    cloth_folding: isEnglish() ? "Cloth folding" : "布料折叠",
  };
  select.innerHTML = [
    `<option value="">${isEnglish() ? "Auto-route implemented task" : "自动匹配已实现任务"}</option>`,
    ...taskCatalog.map((task) => `<option value="${task.task_id}">${labels[task.task_id] || task.title || task.task_id}</option>`),
    `<option value="__other__">${isEnglish() ? "Other / custom task" : "其他 / 自定义任务"}</option>`,
  ].join("");
  if ([...select.options].some((opt) => opt.value === current)) {
    select.value = current;
  }
  updateCustomTaskVisibility();
}

function updateCustomTaskVisibility() {
  const wrap = $("customTaskWrap");
  if (!wrap) return;
  wrap.classList.toggle("is-hidden", ($("taskSelect")?.value || "") !== "__other__");
}

async function fetchAssetRegistry(endpoint) {
  try {
    const resp = await fetch(endpoint.replace(/\/api\/run_experiments$/, "/api/asset_registry"));
    if (!resp.ok) return;
    const data = await resp.json();
    if (Array.isArray(data.robot_assets)) {
      assetRegistry = data;
      renderAssetRegistry();
    }
  } catch {
    renderAssetRegistry();
  }
}

async function fetchRobotCatalog(endpoint) {
  try {
    const resp = await fetch(endpoint.replace(/\/api\/run_experiments$/, "/api/robots"));
    if (!resp.ok) return;
    const data = await resp.json();
    if (Array.isArray(data.robots)) {
      robotCatalog = data.robots;
      renderRobotSelect();
    }
  } catch {
    // Keep empty robot list when API is offline.
  }
}

function renderRobotSelect() {
  const select = $("robotSelect");
  if (!select) return;
  const current = select.value;
  select.innerHTML = [
    `<option value="">${isEnglish() ? "Auto select" : "自动选择"}</option>`,
    ...robotCatalog.map((r) => {
      const ee = r.has_gripper ? ` · ${r.end_effector_name || r.gripper_type || "gripper"}` : "";
      return `<option value="${r.robot_id}">${r.name} (${r.robot_id}) · ${r.dof}DOF${ee}</option>`;
    }),
  ].join("");
  if ([...select.options].some((opt) => opt.value === current)) {
    select.value = current;
  }
  updateRobotInfo();
}

function updateRobotInfo() {
  const wrap = $("robotInfo");
  const text = $("robotInfoText");
  if (!wrap || !text) return;
  const robotId = $("robotSelect")?.value || "";
  if (!robotId) {
    wrap.classList.add("is-hidden");
    return;
  }
  const robot = robotCatalog.find((r) => r.robot_id === robotId);
  if (!robot) {
    wrap.classList.add("is-hidden");
    return;
  }
  wrap.classList.remove("is-hidden");
  const parts = [
    `DOF: ${robot.dof}`,
    robot.has_gripper ? `末端执行器: ${robot.end_effector_name || robot.gripper_type}` : "末端执行器: 无",
    robot.has_gripper ? `夹爪关节: ${(robot.gripper_joint_names || []).join(", ") || "已接入"}` : "",
    `执行器: ${robot.actuator_type}`,
    `厂商: ${robot.manufacturer}`,
    `EE site: ${robot.end_effector_site}`,
  ].filter(Boolean);
  text.textContent = parts.join(" | ");
}

async function fetchNlpRun(goal, limit) {
  const baseEndpoint = $("apiEndpoint").value.trim().replace(/\/api\/run_experiments$/, "");
  const endpoint = `${baseEndpoint}/api/nlp/run`;
  const robotId = $("robotSelect")?.value || "";
  const body = {
    goal,
    limit,
    language: outputLanguage(),
  };
  if (robotId) body.robot_id = robotId;
  const response = await fetch(endpoint, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!response.ok) throw new Error(`NLP API ${response.status}: ${await response.text()}`);
  return await response.json();
}

function displayParsedTaskSpec(taskSpec) {
  const wrap = $("parsedTaskSpec");
  const pre = $("parsedTaskSpecJson");
  if (!wrap || !pre) return;
  if (!taskSpec) {
    wrap.classList.add("is-hidden");
    return;
  }
  wrap.classList.remove("is-hidden");
  pre.textContent = compactTaskDescription(taskSpec);
}

function meters(value) {
  if (value === null || value === undefined || value === "") return "--";
  return `${(Number(value) * 100).toFixed(1)} cm`;
}

function renderAgentPipeline(payload) {
  const target = $("agentPipeline");
  if (!target) return;
  if (!payload) {
    target.classList.add("is-hidden");
    target.innerHTML = "";
    return;
  }
  const steps = [
    ["TaskSpec", payload.standard_task_spec ? "ok" : "pending"],
    ["ExperimentPlan", payload.experiment_plan ? "ok" : "pending"],
    ["SkillPlan", payload.skill_plan ? "ok" : "pending"],
    ["Capability", payload.capability_report ? "ok" : "pending"],
    ["MuJoCo", payload.runs?.length ? "ok" : "pending"],
    ["Evaluation", payload.evaluation_report ? "ok" : "pending"],
    ["RetryPlan", payload.retry_plan ? "ok" : "pending"],
    ["AutoRetry", payload.retry_execution ? "ok" : "pending"],
    ["Memory", payload.experiment_memory?.recorded ? "ok" : "pending"],
  ];
  const robot = payload.capability_report?.robot_selection;
  const tool = payload.capability_report?.tool_selection;
  const robotSpec = payload.robot_spec || {};
  const eeName = robotSpec.end_effector_name || (robotSpec.has_gripper ? robotSpec.gripper_type : "none");
  target.classList.remove("is-hidden");
  target.innerHTML = `
    <h4>通用机械臂实验智能体闭环</h4>
    <div class="pipeline-steps">
      ${steps.map(([name, state]) => `<span class="${state}">${name}</span>`).join("")}
    </div>
    <div class="capability-lines">
      <span><b>机器人选择</b>${robot?.selected_robot_id || payload.robot_id || "auto"} · ${robot?.executable === false ? "能力不足" : "可执行"}</span>
      <span><b>末端执行器</b>${eeName}${robotSpec.has_gripper ? ` · 关节 ${(robotSpec.gripper_joint_names || []).join(", ") || "finger_joint1/finger_joint2"} · actuator ${(robotSpec.gripper_actuator_names || []).join(", ") || "actuator8"}` : ""}</span>
      <span><b>工具选择</b>${tool?.selected_tool_id || "无工具"} ${tool?.reason ? `· ${tool.reason}` : ""}</span>
      <span><b>真实物理</b>MuJoCo 刚体/接触/摩擦/质量/关节执行；桌面穿透会被标记为失败</span>
    </div>
  `;
}

function renderEvaluationPanel(payload) {
  const target = $("evaluationPanel");
  if (!target) return;
  const report = payload?.evaluation_report;
  const taskSpec = payload?.task_spec || {};
  if (!report) {
    target.classList.add("is-hidden");
    target.innerHTML = "";
    return;
  }
  const criteria = taskSpec.success_criteria || {};
  const failures = report.failure_reasons || [];
  const retry = payload.retry_execution;
  const retryComparison = retry?.comparison || {};
  target.classList.remove("is-hidden");
  target.innerHTML = `
    <div>
      <h4>严格评估口径</h4>
      <p>目标位移: ${meters(criteria.target_displacement_m)} · 容差: ${meters(criteria.tolerance)} · 样本数: ${report.sample_count}</p>
    </div>
    <div>
      <h4>失败归因</h4>
      <p>${failures.length ? failures.map((f) => `${f.failure_type}:${f.count}`).join(" / ") : "none"}</p>
    </div>
    <div>
      <h4>下一轮动作</h4>
      <p>${(payload.retry_plan?.changes || report.next_actions || []).join(" / ") || "扩大扰动范围再验证"}</p>
    </div>
    <div>
      <h4>自动闭环重试</h4>
      <p>${retry?.attempted ? `已执行 · 成功率 ${((retryComparison.before_success_rate || 0) * 100).toFixed(1)}% -> ${((retryComparison.after_success_rate || 0) * 100).toFixed(1)}% · 原失败 ${((retryComparison.source_failure_before_rate || 0) * 100).toFixed(1)}% -> ${((retryComparison.source_failure_after_rate || 0) * 100).toFixed(1)}%` : (retry?.reason || "未触发")}</p>
    </div>
    <div>
      <h4>实验记忆</h4>
      <p>${payload.experiment_memory?.recorded ? `已写入 · 历史记录 ${payload.experiment_memory?.summary?.total_records || 0} 条` : "未写入"}</p>
    </div>
  `;
}

function renderClosedLoopPanel(payload) {
  const target = $("closedLoopPanel");
  if (!target) return;
  const retryPlan = payload?.retry_plan;
  const retry = payload?.retry_execution;
  if (!retryPlan && !retry) {
    target.className = "closed-loop-panel is-empty";
    target.innerHTML = `<div class="empty-state">运行 NLP 管线后展示 RetryPlan、自动重试结果和前后对比。</div>`;
    return;
  }

  const comparison = retry?.comparison || {};
  const changes = retryPlan?.changes || retry?.changes || [];
  const beforeRate = comparison.before_success_rate;
  const afterRate = comparison.after_success_rate;
  const beforeFailure = comparison.source_failure_before_rate;
  const afterFailure = comparison.source_failure_after_rate;

  target.className = "closed-loop-panel";
  target.innerHTML = `
    <div class="loop-summary">
      <div>
        <span>RetryPlan</span>
        <strong>${retryPlan?.should_retry ? "需要重试" : "无需重试"}</strong>
        <p>${retryPlan?.reason || retry?.reason || "当前样本未暴露需要修正的失败。"}</p>
      </div>
      <div>
        <span>触发失败类型</span>
        <strong>${retryPlan?.dominant_failure || retry?.source_failure || "none"}</strong>
        <p>${changes.length ? changes.join(" / ") : "未产生参数修正"}</p>
      </div>
      <div>
        <span>AutoRetry</span>
        <strong>${retry?.attempted ? "已执行" : "未触发"}</strong>
        <p>${retry?.runs?.length ? `${retry.runs.length} 组修正实验` : "等待失败归因触发"}</p>
      </div>
    </div>
    <div class="loop-compare">
      <div>
        <span>重试前成功率</span>
        <strong>${beforeRate === undefined ? "--" : pct(beforeRate)}</strong>
      </div>
      <div>
        <span>重试后成功率</span>
        <strong>${afterRate === undefined ? "--" : pct(afterRate)}</strong>
      </div>
      <div>
        <span>原失败占比变化</span>
        <strong>${beforeFailure === undefined ? "--" : `${pct(beforeFailure)} -> ${pct(afterFailure || 0)}`}</strong>
      </div>
    </div>
  `;
}

function renderMemoryPanel(payload) {
  const target = $("memoryPanel");
  if (!target) return;
  const memory = payload?.experiment_memory;
  if (!memory) {
    target.className = "memory-panel is-empty";
    target.innerHTML = `<div class="empty-state">运行 NLP 管线后展示 ExperimentStore 写入状态、历史记录数量和失败记忆统计。</div>`;
    return;
  }

  const summary = memory.summary || {};
  const taskCounts = Object.entries(summary.task_counts || {});
  const failureCounts = Object.entries(summary.failure_counts || {}).sort((a, b) => b[1] - a[1]);
  const bestByTask = Object.entries(summary.best_by_task || {});

  target.className = "memory-panel";
  target.innerHTML = `
    <div class="memory-head">
      <div>
        <span>写入状态</span>
        <strong>${memory.recorded ? "已写入 ExperimentStore" : "未写入"}</strong>
        <p>${memory.recorded_at ? `记录时间: ${memory.recorded_at}` : "没有记录时间"}</p>
      </div>
      <div>
        <span>历史记录</span>
        <strong>${summary.total_records || 0} 条</strong>
        <p>用于后续检索失败模式、最佳结果和参数边界。</p>
      </div>
    </div>
    <div class="memory-grid">
      <div>
        <h4>任务类型统计</h4>
        <p>${taskCounts.length ? taskCounts.map(([k, v]) => `${k}: ${v}`).join(" / ") : "暂无"}</p>
      </div>
      <div>
        <h4>失败记忆</h4>
        <p>${failureCounts.length ? failureCounts.slice(0, 4).map(([k, v]) => `${k}: ${v}`).join(" / ") : "暂无失败记录"}</p>
      </div>
      <div>
        <h4>最佳历史结果</h4>
        <p>${bestByTask.length ? bestByTask.slice(0, 3).map(([k, v]) => `${k}: ${pct(v.success_rate || 0)}`).join(" / ") : "暂无"}</p>
      </div>
    </div>
  `;
}

function applyNlpPayload(payload) {
  currentNlpResult = payload;
  const taskSpec = payload.task_spec;
  if (taskSpec) {
    currentTask = {
      task_id: taskSpec.task_type || "nlp_generated",
      title: taskTypeLabel(taskSpec.task_type),
      description: compactTaskDescription(taskSpec, payload),
      experiment_space: taskSpec.experiment_variables || {},
      execution_kind: "robot_arm_skill_simulation",
      manipulation_actor: payload.robot_spec?.name || payload.robot_id || taskSpec.robot_preference || "auto",
    };
    experimentSpace = taskSpec.experiment_variables || {};
    hypotheses = buildNlpHypotheses(taskSpec);
  }
  if (payload.design) {
    currentAgentDesign = payload.design;
    if (Array.isArray(payload.design.hypotheses) && payload.design.hypotheses.length) {
      hypotheses = payload.design.hypotheses.map((h, index) => ({
        id: h.id || `H${index + 1}`,
        title: h.title || "实验假设",
        text: h.claim || h.text || "",
        metric: h.metric || "success_rate",
      }));
    }
    if (payload.design.experiment_space) {
      experimentSpace = payload.design.experiment_space;
    }
  }
  if (payload.analysis) {
    currentAgentAnalysis = payload.analysis;
  }
  displayParsedTaskSpec(taskSpec);
  renderAgentPipeline(payload);
  renderPlan();
}

function renderNlpMetrics(summary) {
  $("mRuns").textContent = summary.num_runs || summary.numRuns || 0;
  const rate = summary.success_rate ?? summary.successRate ?? 0;
  $("mSuccess").textContent = pct(rate);
  const mainFail = summary.main_failure_type || summary.mainFailure?.[0] || "none";
  $("mFailure").textContent = mainFail;
  const explainRate = summary.explainable_rate ?? summary.explainableRate ?? 0;
  $("mExplain").textContent = pct(explainRate);
}

function computeNlpVariableImpact(runs, space) {
  const impact = {};
  for (const key of Object.keys(space)) {
    const groups = {};
    for (const run of runs) {
      const val = String(run[key] ?? "unknown");
      if (!groups[val]) groups[val] = { total: 0, success: 0 };
      groups[val].total++;
      if (run.success) groups[val].success++;
    }
    impact[key] = {};
    for (const [val, g] of Object.entries(groups)) {
      impact[key][val] = g.success / Math.max(1, g.total);
    }
  }
  return impact;
}

function renderNlpResults(payload) {
  const runs = payload.runs || [];
  const summary = payload.summary || {};
  const analysis = payload.analysis || {};

  currentRuns = runs;

  // Compute variable impact from runs
  const varImpact = computeNlpVariableImpact(runs, experimentSpace);

  // Compute main failure from distribution
  const failDist = summary.failure_distribution || {};
  const mainFailEntry = Object.entries(failDist).sort((a, b) => b[1] - a[1])[0] || ["none", 0];

  currentSummary = {
    numRuns: summary.num_runs || runs.length,
    successRate: summary.success_rate || 0,
    failureDistribution: failDist,
    mainFailure: mainFailEntry,
    explainableRate: summary.explainable_rate ?? 0,
    variableKeys: Object.keys(experimentSpace),
    byVariables: {},
    findings: analysis.findings || [],
    recommendations: analysis.recommendations || [],
  };

  renderNlpMetrics(summary);
  renderEvaluationPanel(payload);
  renderClosedLoopPanel(payload);
  renderMemoryPanel(payload);

  // Failure distribution chart
  renderBarChart(
    "failureChart",
    Object.entries(failDist).sort((a, b) => b[1] - a[1]).map(([label, value]) => ({ label, value })),
    { failure: true, percent: true }
  );

  // Variable impact chart
  const variableItems = Object.entries(varImpact).flatMap(([key, groups]) =>
    Object.entries(groups).map(([label, rate]) => ({
      label: `${key}:${label}`,
      value: rate,
    }))
  );
  renderBarChart("variableChart", variableItems, { percent: true });

  // Analysis findings and recommendations
  if (analysis.findings) {
    currentSummary.findings = analysis.findings.map((f) => ({
      title: f.title || "发现",
      body: f.body || f.evidence || "",
      confidence: Number(f.confidence ?? 0.75),
    }));
  }
  if (analysis.recommendations) {
    currentSummary.recommendations = analysis.recommendations.map((r) => ({
      title: r.title || "建议",
      body: r.body || "",
    }));
  }
  renderAnalysis(currentSummary);
  renderLogs(runs);

  // Demo trace / replay
  if (payload.demo_trace?.replays?.length || payload.demo_trace?.image_frames?.length) {
    renderReplay(payload.demo_trace);
  } else {
    clearReplay("NLP 管线未返回渲染帧。");
  }
}

function renderAssetRegistry() {
  const robotTarget = $("robotAssets");
  const blueprintTarget = $("taskBlueprints");
  if (!robotTarget || !blueprintTarget) return;
  const robots = assetRegistry.robot_assets || [];
  const blueprints = assetRegistry.task_blueprints || [];
  robotTarget.innerHTML = robots.length
    ? robots.map((asset) => `
      <article class="asset-item">
        <strong>${asset.name}</strong>
        <p>${asset.category} · ${asset.asset_id}</p>
        <small class="${asset.available ? "ok" : "bad"}">${asset.available ? (isEnglish() ? "downloaded" : "已下载") : (isEnglish() ? "missing" : "未下载")}</small>
      </article>
    `).join("")
    : `<p class="muted">${isEnglish() ? "Start the backend to load the asset registry." : "启动后端后加载资产库。"}</p>`;
  blueprintTarget.innerHTML = blueprints.length
    ? blueprints.map((bp) => `
      <article class="asset-item">
        <strong>${isEnglish() ? bp.name_en : bp.name_zh}</strong>
        <p>${bp.required_assets.join(", ")}</p>
        <small class="${bp.status === "implemented" ? "ok" : "planned"}">${
          bp.status === "implemented"
            ? (isEnglish() ? "implemented" : "已实现")
            : bp.status === "assets_downloaded"
              ? (isEnglish() ? "assets downloaded, runner pending" : "资产已下载，runner 待实现")
              : (isEnglish() ? "planned" : "规划中")
        }</small>
      </article>
    `).join("")
    : `<p class="muted">${isEnglish() ? "No capability components loaded." : "未加载能力组件。"}</p>`;
}

function renderEmptyPlan() {
  $("hypotheses").innerHTML = "";
  $("variables").innerHTML = "";
}

function renderPlan() {
  $("hypotheses").innerHTML = hypotheses.map((h) => `
    <article class="info-card">
      <strong>${h.id} ${h.title}</strong>
      <p>${h.text}</p>
      <p>${isEnglish() ? "Metric" : "观察指标"}：${h.metric}</p>
    </article>
  `).join("");

  const supportedIds = new Set(currentTask.supported_objects || []);
  const taskObjects = objectLibrary.filter((obj) => !supportedIds.size || supportedIds.has(obj.object_id));
  const boundaryText = currentNlpResult
    ? (isEnglish()
      ? "NLP dynamic pipeline: parses the goal, composes a MuJoCo scene, runs experiments, then recommends the next iteration."
      : "NLP 动态管线：解析目标、组装 MuJoCo 场景、运行实验，并给出下一轮方向。")
    : String(currentTask.execution_kind || "").startsWith("robot_arm")
    ? (isEnglish()
      ? "Scripted MuJoCo robot-arm experiment; not policy learning or real hardware."
      : "脚本化 MuJoCo 机械臂实验；不是策略训练或真实硬件控制。")
    : (isEnglish()
      ? "MuJoCo proxy task with task-specific actuators; not a full robot-arm trajectory."
      : "MuJoCo 代理任务；不是完整机械臂运动轨迹。");
  const taskCard = `
    <div class="task-card">
      <div class="task-card-head">
        <strong>${currentTask.title || currentTask.task_id || "动态任务"}</strong>
        <span>${taskExecutionLabel()}</span>
      </div>
      <p>${currentTask.description || taskScopeLabel()}</p>
      <div class="task-meta">
        <span><b>${isEnglish() ? "Executor" : "执行器"}</b>${currentTask.manipulation_actor || taskExecutionLabel()}</span>
        <span><b>${isEnglish() ? "Objects" : "物体"}</b>${compactObjectLine(taskObjects)}</span>
      </div>
      <small>${boundaryText}</small>
    </div>
  `;

  $("variables").innerHTML = taskCard + Object.entries(experimentSpace).map(([name, values]) => `
    <div class="var-row">
      <strong>${name}</strong>
      <div class="tagline">${values.map((v) => `<span class="tag">${v}</span>`).join("")}</div>
    </div>
  `).join("");

  $("runStatus").textContent = `已生成实验设计 · ${currentTask.title || currentTask.task_id}`;
}

function seededNoise(seed) {
  const x = Math.sin(seed * 12.9898) * 43758.5453;
  return x - Math.floor(x);
}

function scoreRun(run, idx) {
  let score = 0.72;
  if (run.friction === "low") score -= 0.50;
  if (run.friction === "high") score += 0.08;
  if (run.grasp_height_offset === "-2cm") score -= 0.50;
  if (run.grasp_height_offset === "+2cm") score -= 0.02;
  if (run.object_offset === "medium") score -= 0.05;
  if (run.object_offset === "large") score -= 0.15;
  if (run.vision_noise === "light") score -= 0.04;
  if (run.vision_noise === "heavy") score -= 0.10;
  if (run.object_offset === "large" && run.vision_noise === "heavy") score -= 0.08;
  score += (seededNoise(idx + 17) - 0.5) * 0.14;
  return Math.max(0.02, Math.min(0.96, score));
}

function classifyFailure(run, idx) {
  if (run.friction === "low") return "slip";
  if (run.grasp_height_offset === "-2cm") return "grasp_miss";
  if (run.object_offset === "large" && run.vision_noise === "heavy") return "grasp_miss";
  if (run.friction === "medium" && seededNoise(idx + 3) > 0.6) return "slip";
  return "grasp_miss";
}

function buildRuns(limit) {
  const rows = [];
  let idx = 1;
  const objectOffsets = experimentSpace.object_offset || ["small", "medium", "large"];
  const frictions = experimentSpace.friction || ["low", "medium", "high"];
  const heights = experimentSpace.grasp_height_offset || ["-2cm", "0", "+2cm"];
  const noises = experimentSpace.vision_noise || ["none", "light", "heavy"];
  const pushRow = (object_offset, friction, grasp_height_offset, vision_noise) => {
    const run = {
      run_id: `exp_${String(idx).padStart(3, "0")}`,
      object_offset,
      friction,
      grasp_height_offset,
      vision_noise,
      control_freq: "normal"
    };
    const score = scoreRun(run, idx);
    const success = seededNoise(idx + 101) < score;
    const failure_type = success ? "none" : classifyFailure(run, idx);
    rows.push({
      ...run,
      success,
      failure_type,
      trajectory_error: Number((0.025 + (1 - score) * 0.24 + seededNoise(idx + 7) * 0.03).toFixed(3)),
      collision_count: failure_type === "collision" ? 1 + Math.floor(seededNoise(idx + 41) * 3) : 0,
      final_distance: Number((success ? 0.02 + seededNoise(idx + 5) * 0.04 : 0.09 + seededNoise(idx + 6) * 0.18).toFixed(3)),
      max_grip_force: success ? Number((0.5 + seededNoise(idx + 81) * 2.0).toFixed(3)) : Number((seededNoise(idx + 82) * 0.3).toFixed(3)),
      cube_slip_detected: failure_type === "slip"
    });
    idx += 1;
  };

  if (limit === 27) {
    for (const object_offset of objectOffsets) {
      for (const friction of frictions) {
        for (const grasp_height_offset of heights) {
          pushRow(object_offset, friction, grasp_height_offset, noises[(idx - 1) % noises.length]);
        }
      }
    }
    return rows;
  }

  for (const object_offset of objectOffsets) {
    for (const friction of frictions) {
      for (const grasp_height_offset of heights) {
        for (const vision_noise of noises) pushRow(object_offset, friction, grasp_height_offset, vision_noise);
      }
    }
  }
  return rows.slice(0, limit);
}

function summarize(rows) {
  const successCount = rows.filter((r) => r.success).length;
  const failures = rows.filter((r) => !r.success);
  const failureDistribution = Object.fromEntries(
    Object.entries(groupBy(failures, "failure_type")).map(([key, value]) => [key, value.length / Math.max(1, rows.length)])
  );
  const mainFailure = Object.entries(failureDistribution).sort((a, b) => b[1] - a[1])[0] || ["none", 0];
  const variableKeys = inferVariableKeys(rows);
  const byVariables = Object.fromEntries(variableKeys.map((key) => [key, groupBy(rows, key)]));

  const findings = variableKeys.slice(0, 3).map((key) => {
    const groups = Object.entries(byVariables[key]).map(([label, items]) => ({
      label,
      rate: items.filter((r) => r.success).length / Math.max(1, items.length)
    })).sort((a, b) => a.rate - b.rate);
    const low = groups[0];
    const high = groups[groups.length - 1] || low;
    return {
      title: `${key} 对成功率有可观测影响`,
      body: `${key}=${low?.label ?? "unknown"} 的成功率为 ${pct(low?.rate ?? 0)}，${key}=${high?.label ?? "unknown"} 的成功率为 ${pct(high?.rate ?? 0)}。`,
      confidence: Math.min(0.9, Math.max(0.55, Math.abs((high?.rate ?? 0) - (low?.rate ?? 0)) + 0.55))
    };
  });

  return {
    numRuns: rows.length,
    successRate: successCount / Math.max(1, rows.length),
    failureDistribution,
    mainFailure,
    explainableRate: failures.length ? failures.filter((r) => r.failure_type !== "timeout").length / failures.length : 1,
    variableKeys,
    byVariables,
    findings,
    recommendations: [
      {
        title: "收窄变量空间",
        body: "下一轮优先固定成功率差异较小的变量，把预算集中到失败类型最集中的变量组合上。"
      },
      {
        title: "补充边界样本",
        body: "对主要失败类型附近的配置增加重复实验，区分偶然扰动和稳定物理机制。"
      }
    ]
  };
}

function renderMetrics(summary) {
  $("mRuns").textContent = summary.numRuns;
  $("mSuccess").textContent = pct(summary.successRate);
  $("mFailure").textContent = summary.mainFailure[0];
  $("mExplain").textContent = pct(summary.explainableRate);
}

function renderBarChart(target, items, options = {}) {
  const max = Math.max(...items.map((item) => item.value), 0.01);
  $(target).innerHTML = items.map((item) => `
    <div class="bar-row">
      <span>${item.label}</span>
      <div class="bar-track">
        <div class="bar-fill ${options.failure ? "failure" : ""}" style="width:${Math.round(item.value / max * 100)}%"></div>
      </div>
      <strong>${options.percent ? pct(item.value) : Number(item.value).toFixed(2)}</strong>
    </div>
  `).join("");
}

function renderAnalysis(summary) {
  $("findings").innerHTML = summary.findings.map((item) => `
    <article class="info-card">
      <strong>${item.title} · 置信度 ${pct(item.confidence ?? 0.75)}</strong>
      <p>${item.body}</p>
    </article>
  `).join("");

  $("recommendations").innerHTML = summary.recommendations.map((item) => `
    <article class="info-card">
      <strong>${item.title}</strong>
      <p>${item.body}</p>
    </article>
  `).join("");
}

function formatValue(value) {
  if (typeof value === "number") return Number.isInteger(value) ? String(value) : value.toFixed(3);
  if (typeof value === "boolean") return value ? "Y" : "";
  return value ?? "";
}

function renderLogs(rows) {
  const variableKeys = inferVariableKeys(rows).slice(0, 5);
  const extraMetrics = ["success", "failure_type", "final_distance", "trajectory_error"];
  if (rows.some((r) => "target_displacement_m" in r)) extraMetrics.push("target_displacement_m");
  if (rows.some((r) => "object_displacement" in r)) extraMetrics.push("object_displacement");
  if (rows.some((r) => "displacement_error_m" in r)) extraMetrics.push("displacement_error_m");
  if (rows.some((r) => "table_clearance_ok" in r)) extraMetrics.push("table_clearance_ok");
  if (rows.some((r) => "min_ee_z" in r)) extraMetrics.push("min_ee_z");
  if (rows.some((r) => "table_contact_steps" in r)) extraMetrics.push("table_contact_steps");
  if (rows.some((r) => "table_contact_pairs" in r)) extraMetrics.push("table_contact_pairs");
  if (rows.some((r) => "max_grip_force" in r)) extraMetrics.push("max_grip_force");
  if (rows.some((r) => "max_push_force" in r)) extraMetrics.push("max_push_force");
  if (rows.some((r) => "contact_steps" in r)) extraMetrics.push("contact_steps");
  const columns = ["run_id", ...variableKeys, ...extraMetrics];
  const table = $("logRows").closest("table");
  table.querySelector("thead tr").innerHTML = columns.map((key) => `<th>${key}</th>`).join("");
  $("logRows").innerHTML = rows.slice(0, 14).map((row) => `
    <tr>
      ${columns.map((key) => `<td class="${key === "success" ? (row.success ? "ok" : "bad") : ""}">${formatValue(row[key])}</td>`).join("")}
    </tr>
  `).join("");
}

async function fetchApiRuns(limit) {
  const endpoint = $("apiEndpoint").value.trim();
  const response = await fetch(endpoint, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      goal: effectiveGoal(),
      task_id: selectedTaskId(),
      limit,
      language: outputLanguage()
    })
  });
  if (!response.ok) throw new Error(`MuJoCo API ${response.status}: ${await response.text()}`);
  const payload = await response.json();
  if (!Array.isArray(payload.runs)) throw new Error("MuJoCo API response missing runs[]");
  return payload;
}

async function fetchAgentRuns(limit) {
  const endpoint = $("apiEndpoint").value.trim().replace(/\/api\/run_experiments$/, "/api/agent/run");
  const response = await fetch(endpoint, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      goal: effectiveGoal(),
      task_id: selectedTaskId(),
      limit,
      language: outputLanguage()
    })
  });
  if (!response.ok) throw new Error(`Agent API ${response.status}: ${await response.text()}`);
  const payload = await response.json();
  if (!Array.isArray(payload.runs)) throw new Error("Agent API response missing runs[]");
  return payload;
}

function applyAgentPayload(payload) {
  currentTask = payload.task || currentTask;
  if (Array.isArray(payload.object_library)) {
    objectLibrary = payload.object_library;
  }
  if (payload.asset_registry) {
    assetRegistry = payload.asset_registry;
    renderAssetRegistry();
  }
  experimentSpace = payload.design?.experiment_space || payload.task?.experiment_space || experimentSpace;
  currentAgentDesign = payload.design || null;
  currentAgentAnalysis = payload.analysis || null;
  currentAgentTrace = payload.agent_trace || [];
  currentAgentApiCalls = payload.model_api_calls || null;
  currentExecutionPlan = payload.execution_plan || null;
  currentAgentProvider = {
    provider: payload.agent_provider || "",
    protocol: payload.protocol || "",
    model: payload.model || "",
    source: payload.source || ""
  };
  if (Array.isArray(currentAgentDesign?.hypotheses) && currentAgentDesign.hypotheses.length) {
    hypotheses = currentAgentDesign.hypotheses.map((h, index) => ({
      id: h.id || `H${index + 1}`,
      title: h.title || "实验假设",
      text: h.claim || h.text || "",
      metric: h.metric || "success_rate"
    }));
  }
  renderPlan();
}

async function runExperiment() {
  const limit = Number($("sampleSize").value);
  const useAgent = $("useApi").checked && $("useAgent").checked;
  if ($("runBtn").disabled) return;
  if (useAgent && limit === 81) {
    showToast("智能体 API 的 81 组会明显更慢：需要多次模型调用加完整 MuJoCo 批量仿真。建议先用 27 组确认任务路由和变量空间。", 7000);
  }
  const feedbackMode = useAgent ? "agent" : ($("useApi").checked ? "mujoco" : "static");
  if (feedbackMode !== "static") {
    startRunFeedback(feedbackMode, limit);
    await sleepFrame();
  } else {
    $("runStatus").textContent = "正在运行内置评测...";
  }
  try {
    currentAgentDesign = null;
    currentAgentAnalysis = null;
    currentAgentTrace = [];
    currentAgentApiCalls = null;
    currentExecutionPlan = null;
    currentAgentProvider = null;
    if (useAgent) {
      const payload = await fetchAgentRuns(limit);
      applyAgentPayload(payload);
      currentRuns = payload.runs;
      currentTrace = payload.demo_trace || null;
      $("runStatus").textContent = `已完成智能体评测 · ${payload.source}`;
    } else if ($("useApi").checked) {
      const endpoint = $("apiEndpoint").value.trim();
      await fetchExperimentSpace(endpoint);
      const payload = await fetchApiRuns(limit);
      currentRuns = payload.runs;
      currentTrace = payload.demo_trace || null;
      if (payload.task) {
        currentTask = payload.task;
        experimentSpace = payload.task.experiment_space || experimentSpace;
      }
      $("runStatus").textContent = `已完成评测 · ${payload.source || "mujoco"}`;
    } else {
      currentTask = {
        task_id: "fr3_pick_place",
        title: "Franka FR3 pick-and-place",
        description: "当前默认任务：FR3 抓取并放置物块。",
        experiment_space: experimentSpace
      };
      currentRuns = buildRuns(limit);
      currentTrace = null;
      $("runStatus").textContent = "已完成评测 · static";
    }
  } catch (error) {
    $("runStatus").textContent = "API 调用失败，已切回内置评测";
    showToast(`API 调用失败：${error.message}。已自动回退到内置评测数据。`);
    console.error(error);
    currentRuns = buildRuns(limit);
    currentTrace = null;
  } finally {
    if (feedbackMode !== "static") {
      stopRunFeedback();
    }
  }

  currentSummary = summarize(currentRuns);
  if (currentAgentAnalysis) {
    if (Array.isArray(currentAgentAnalysis.findings)) {
      currentSummary.findings = currentAgentAnalysis.findings.map((item) => ({
        title: item.title || "智能体发现",
        body: item.body || item.evidence || "",
        confidence: Number(item.confidence ?? 0.75)
      }));
    }
    if (Array.isArray(currentAgentAnalysis.recommendations)) {
      currentSummary.recommendations = currentAgentAnalysis.recommendations.map((item) => ({
        title: item.title || "下一轮实验",
        body: item.body || "",
      }));
    }
  }

  renderMetrics(currentSummary);
  renderBarChart(
    "failureChart",
    Object.entries(currentSummary.failureDistribution).sort((a, b) => b[1] - a[1]).map(([label, value]) => ({ label, value })),
    { failure: true, percent: true }
  );

  const variableItems = currentSummary.variableKeys.slice(0, 2).flatMap((key) =>
    Object.entries(currentSummary.byVariables[key] || {}).map(([label, rows]) => ({
      label: `${key}:${label}`,
      value: rows.filter((r) => r.success).length / Math.max(1, rows.length)
    }))
  );
  renderBarChart("variableChart", variableItems, { percent: true });
  renderAnalysis(currentSummary);
  renderLogs(currentRuns);

  if ((currentTrace?.replays?.length || currentTrace?.image_frames?.length) && currentTaskUsesRobotArm()) {
    renderReplay(currentTrace);
  } else if (currentTrace?.replays?.length || currentTrace?.image_frames?.length) {
    clearReplay("该任务返回的是代理场景渲染，不是机械臂 MuJoCo 回放。");
  } else {
    clearReplay($("useApi").checked ? "MuJoCo API 未返回真实渲染帧。" : "当前为内置数据评测模式，不展示 MuJoCo 渲染回放。");
  }
  makeReport();
}

function lerp(a, b, t) {
  return a + (b - a) * t;
}

function lerpPoint(a, b, t) {
  return { x: lerp(a.x, b.x, t), y: lerp(a.y || 0, b.y || 0, t), z: lerp(a.z, b.z, t) };
}

function interpolateFrames(frames, stepsPerSegment = 12) {
  if (!frames || frames.length < 2) return frames || [];
  const expanded = [];
  for (let i = 0; i < frames.length - 1; i += 1) {
    const a = frames[i];
    const b = frames[i + 1];
    for (let s = 0; s < stepsPerSegment; s += 1) {
      const t = s / stepsPerSegment;
      expanded.push({
        label: t < 0.5 ? a.label : b.label,
        time: Number(lerp(a.time || 0, b.time || 0, t).toFixed(3)),
        gripper: { ...lerpPoint(a.gripper, b.gripper, t), closed: lerp(a.gripper.closed || 0, b.gripper.closed || 0, t) },
        cube: lerpPoint(a.cube, b.cube, t),
        target: a.target || b.target
      });
    }
  }
  expanded.push(frames[frames.length - 1]);
  return expanded;
}

function setWorldBounds(frames) {
  let xMin = Infinity, xMax = -Infinity, zMin = Infinity, zMax = -Infinity;
  for (const frame of frames || []) {
    for (const obj of [frame.gripper, frame.cube, frame.target]) {
      if (!obj) continue;
      xMin = Math.min(xMin, obj.x);
      xMax = Math.max(xMax, obj.x);
      zMin = Math.min(zMin, obj.z);
      zMax = Math.max(zMax, obj.z);
    }
  }
  const xPad = (xMax - xMin) * 0.18 || 0.05;
  const zPad = (zMax - zMin) * 0.18 || 0.05;
  worldBounds = { xMin: xMin - xPad, xMax: xMax + xPad, zMin: zMin - zPad, zMax: zMax + zPad };
}

function worldToCanvas(point, width, height) {
  const b = worldBounds || { xMin: -0.12, xMax: 0.34, zMin: 0, zMax: 0.18 };
  return {
    x: 76 + ((point.x - b.xMin) / Math.max(0.001, b.xMax - b.xMin)) * (width - 152),
    y: height - 82 - ((point.z - b.zMin) / Math.max(0.001, b.zMax - b.zMin)) * (height - 140)
  };
}

function drawReplayFrame(canvasId, frame, metaText = "") {
  const canvas = $(canvasId);
  const ctx = canvas.getContext("2d");
  const { width, height } = canvas;
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#f8fafc";
  ctx.fillRect(0, 0, width, height);
  ctx.strokeStyle = "#dce3ea";
  ctx.lineWidth = 1;
  for (let x = 80; x < width - 80; x += 70) {
    ctx.beginPath();
    ctx.moveTo(x, 84);
    ctx.lineTo(x, height - 92);
    ctx.stroke();
  }
  ctx.fillStyle = "#d6dde5";
  ctx.fillRect(62, height - 78, width - 124, 18);
  const target = worldToCanvas(frame.target, width, height);
  ctx.fillStyle = "rgba(26,154,108,0.24)";
  ctx.fillRect(target.x - 42, height - 104, 84, 22);
  ctx.fillStyle = "#1a9a6c";
  ctx.fillText("target", target.x - 18, height - 112);

  const cube = worldToCanvas(frame.cube, width, height);
  ctx.fillStyle = "#2f80ed";
  ctx.fillRect(cube.x - 18, cube.y - 18, 36, 36);

  const grip = worldToCanvas(frame.gripper, width, height);
  ctx.strokeStyle = "#22313f";
  ctx.lineWidth = 10;
  ctx.lineCap = "round";
  ctx.beginPath();
  ctx.moveTo(grip.x - 22, grip.y);
  ctx.lineTo(grip.x + 22, grip.y);
  ctx.stroke();
  ctx.fillStyle = "#22313f";
  ctx.fillRect(grip.x - 18, grip.y - 18, 36, 36);

  ctx.fillStyle = "#1d2733";
  ctx.font = "700 18px Segoe UI, Arial";
  ctx.fillText("MuJoCo trajectory replay", 28, 36);
  ctx.font = "14px Segoe UI, Arial";
  ctx.fillStyle = "#637384";
  ctx.fillText(`${frame.label} · t=${frame.time}s ${metaText}`, 28, 62);
}

function clearReplay(message = "等待 MuJoCo 评测运行") {
  if (replayHandle) cancelAnimationFrame(replayHandle);
  replayHandle = null;
  $("replayCard").classList.add("is-empty");
  $("replayMeta").textContent = "运行 NLP 管线后展示真实 MuJoCo 渲染回放。";
  $("replayEmpty").textContent = message;
  $("replayProgressSuccess").style.width = "0";
  $("replayProgressFailure").style.width = "0";
  $("replayStageSuccess").textContent = "等待运行";
  $("replayStageFailure").textContent = "等待运行";
}

function drawImageFrame(canvasId, src, label, metaText = "") {
  const canvas = $(canvasId);
  const ctx = canvas.getContext("2d");
  const image = new Image();
  image.onload = () => {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
    const labelY = canvas.height - 72;
    ctx.fillStyle = "rgba(13, 21, 31, 0.68)";
    ctx.fillRect(18, labelY, 336, 52);
    ctx.fillStyle = "#fff";
    ctx.font = "700 17px Segoe UI, Arial";
    ctx.fillText("MuJoCo render", 34, labelY + 22);
    ctx.font = "13px Segoe UI, Arial";
    ctx.fillText(`${label} ${metaText}`, 34, labelY + 42);
  };
  image.onerror = () => {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = "#f8fafc";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = "#b42318";
    ctx.font = "700 16px Segoe UI, Arial";
    ctx.fillText(isEnglish() ? "Replay frame failed to load" : "回放帧加载失败", 28, 42);
    ctx.font = "13px Segoe UI, Arial";
    ctx.fillText(src, 28, 68, canvas.width - 56);
  };
  image.src = resolveReplaySrc(src);
}

function resolveReplaySrc(src) {
  if (!src || src.startsWith("data:") || /^https?:\/\//i.test(src)) return src;
  if (!src.startsWith("assets/")) return src;
  try {
    const endpoint = $("apiEndpoint")?.value?.trim() || "";
    const url = new URL(endpoint);
    return `${url.origin}/${src}`;
  } catch (_error) {
    return src;
  }
}

function playReplayPair(success, failure) {
  const durationMs = 9000;
  const sImages = success.image_frames || [];
  const fImages = failure.image_frames || [];
  const sLabels = success.labels || sImages.map((_, i) => `success-${i + 1}`);
  const fLabels = failure.labels || fImages.map((_, i) => `failure-${i + 1}`);
  if (replayHandle) cancelAnimationFrame(replayHandle);
  let start = null;
  const play = (ts) => {
    if (!start) start = ts;
    const progress = ((ts - start) % durationMs) / durationMs;
    const sIdx = Math.min(sImages.length - 1, Math.floor(progress * sImages.length));
    const fIdx = Math.min(fImages.length - 1, Math.floor(progress * fImages.length));
    if (sImages[sIdx]) drawImageFrame("mujocoReplaySuccess", sImages[sIdx], sLabels[sIdx], `(${sIdx + 1}/${sImages.length})`);
    if (fImages[fIdx]) drawImageFrame("mujocoReplayFailure", fImages[fIdx], fLabels[fIdx], `(${fIdx + 1}/${fImages.length})`);
    const progressText = `${Math.round(progress * 100)}%`;
    $("replayProgressSuccess").style.width = progressText;
    $("replayProgressFailure").style.width = progressText;
    $("replayStageSuccess").textContent = sLabels[sIdx] || "success";
    $("replayStageFailure").textContent = fLabels[fIdx] || "failure";
    replayHandle = requestAnimationFrame(play);
  };
  replayHandle = requestAnimationFrame(play);
}

function playTrajectoryPair(success, failure) {
  const durationMs = 6000;
  const sFrames = interpolateFrames(success.trajectory || [], 8);
  const fFrames = interpolateFrames(failure.trajectory || [], 8);
  if (!sFrames.length && !fFrames.length) return;
  if (replayHandle) cancelAnimationFrame(replayHandle);
  let start = null;
  const play = (ts) => {
    if (!start) start = ts;
    const progress = ((ts - start) % durationMs) / durationMs;
    if (sFrames.length) {
      const sIdx = Math.min(sFrames.length - 1, Math.floor(progress * sFrames.length));
      drawReplayFrame("mujocoReplaySuccess", sFrames[sIdx], `(${sIdx + 1}/${sFrames.length})`);
      $("replayStageSuccess").textContent = sFrames[sIdx].label || "trajectory";
    }
    if (fFrames.length) {
      const fIdx = Math.min(fFrames.length - 1, Math.floor(progress * fFrames.length));
      drawReplayFrame("mujocoReplayFailure", fFrames[fIdx], `(${fIdx + 1}/${fFrames.length})`);
      $("replayStageFailure").textContent = fFrames[fIdx].label || "trajectory";
    }
    const progressText = `${Math.round(progress * 100)}%`;
    $("replayProgressSuccess").style.width = progressText;
    $("replayProgressFailure").style.width = progressText;
    replayHandle = requestAnimationFrame(play);
  };
  replayHandle = requestAnimationFrame(play);
}

function renderReplay(trace) {
  $("replayCard").classList.remove("is-empty");
  if (trace?.replays?.length) {
    const success = trace.replays[0];
    const failure = trace.replays[1] || trace.replays[0];
    $("replayTitleSuccess").textContent = success.title || "成功样例";
    $("replayTitleFailure").textContent = failure.title || "失败样例";
    $("replayMeta").textContent = `${trace.model || success.model || "MuJoCo model"} · 双样例回放`;
    // Use trajectory data if available, otherwise image frames
    if (success.trajectory?.length || failure.trajectory?.length) {
      setWorldBounds([...(success.trajectory || []), ...(failure.trajectory || [])]);
      playTrajectoryPair(success, failure);
    } else {
      playReplayPair(success, failure);
    }
    return;
  }

  if (trace?.image_frames?.length) {
    const replay = {
      image_frames: trace.image_frames,
      labels: trace.labels || trace.image_frames.map((_, i) => `frame-${i + 1}`),
    };
    $("replayTitleSuccess").textContent = trace.title || "MuJoCo 渲染回放";
    $("replayTitleFailure").textContent = trace.title || "MuJoCo 渲染回放";
    $("replayMeta").textContent = `${trace.model || trace.source || "MuJoCo model"} · 真实仿真渲染帧`;
    playReplayPair(replay, replay);
    return;
  }

  clearReplay("该任务只返回了抽象轨迹点，未返回真实 MuJoCo 机械臂渲染帧。");
  return;
}

function makeReport() {
  if (!currentSummary) {
    $("reportPreview").textContent = isEnglish() ? "Run a simulation experiment first." : "请先运行仿真实验。";
    return;
  }
  const failureLines = Object.entries(currentSummary.failureDistribution).sort((a, b) => b[1] - a[1]).map(([k, v]) => `- ${k}: ${pct(v)}`).join("\n");
  const findingLines = currentSummary.findings.map((f, i) => `${i + 1}. ${f.title}: ${f.body}`).join("\n");
  const recLines = currentSummary.recommendations.map((r, i) => `${i + 1}. ${r.title}: ${r.body}`).join("\n");
  const variableNames = inferVariableKeys(currentRuns).join(", ");
  const agentLines = currentAgentTrace.length
    ? `\n## ${isEnglish() ? "Agent Trace" : "智能体执行轨迹"}\n\n${currentAgentTrace.map((step, i) => `${i + 1}. ${step}`).join("\n")}\n`
    : "";
  const apiCallLine = currentAgentApiCalls
    ? `\n## ${isEnglish() ? "Model API Calls" : "模型 API 调用"}\n\n- ${isEnglish() ? "Call count" : "调用次数"}: ${currentAgentApiCalls.count}\n- ${isEnglish() ? "Stages" : "阶段"}: ${(currentAgentApiCalls.calls || []).map((c) => c.stage).join(", ")}\n- ${currentAgentApiCalls.note || ""}\n`
    : "";
  const agentProviderLine = currentAgentProvider
    ? `\n## ${isEnglish() ? "Agent Provider" : "智能体来源"}\n\n- Provider: ${currentAgentProvider.provider || "unknown"}\n- Protocol: ${currentAgentProvider.protocol || "unknown"}\n- Model: ${currentAgentProvider.model || "unknown"}\n- Source: ${currentAgentProvider.source || ""}\n`
    : "";
  const executionLine = currentExecutionPlan
    ? `\n## ${isEnglish() ? "Execution Plan" : "执行计划"}\n\n- Runner: ${currentExecutionPlan.runner_module || currentExecutionPlan.task_id}\n- ${isEnglish() ? "Real robot-arm scene" : "真实机械臂场景"}: ${currentExecutionPlan.uses_real_robot_arm_scene ? "yes" : "no"}\n- ${isEnglish() ? "Replay policy" : "回放规则"}: ${currentExecutionPlan.replay_policy || ""}\n`
    : "";
  const routeLine = currentAgentDesign || currentAgentAnalysis
    ? `\n- ${isEnglish() ? "Task" : "任务"}: ${currentTask.title || currentTask.task_id}\n`
    : "";
  const capabilityLine = currentAgentDesign?.capability_boundary || currentAgentAnalysis?.capability_boundary || currentAgentDesign?.rationale || "";
  const evaluation = currentNlpResult?.evaluation_report;
  const criteria = currentNlpResult?.task_spec?.success_criteria || {};
  const evalLine = evaluation
    ? `\n## ${isEnglish() ? "Strict Evaluation" : "严格评估口径"}\n\n- ${isEnglish() ? "Target displacement" : "目标位移"}: ${meters(criteria.target_displacement_m)}\n- ${isEnglish() ? "Tolerance" : "容差"}: ${meters(criteria.tolerance)}\n- ${isEnglish() ? "Sample count" : "样本数"}: ${evaluation.sample_count}\n- ${isEnglish() ? "Success count" : "成功数"}: ${evaluation.success_count}\n- ${isEnglish() ? "Failure count" : "失败数"}: ${evaluation.failure_count}\n- ${isEnglish() ? "Physical safety" : "物理安全"}: table_clearance_ok / table_penetration is part of failure attribution\n`
    : "";
  const retryExecution = currentNlpResult?.retry_execution;
  const retryCmp = retryExecution?.comparison || {};
  const retryLine = retryExecution
    ? `\n## ${isEnglish() ? "Auto Retry Loop" : "自动闭环重试"}\n\n- ${isEnglish() ? "Attempted" : "是否执行"}: ${retryExecution.attempted ? "yes" : "no"}\n- ${isEnglish() ? "Source failure" : "触发失败类型"}: ${retryExecution.source_failure || "none"}\n- ${isEnglish() ? "Changes" : "调整动作"}: ${(retryExecution.changes || []).join(", ") || "none"}\n- ${isEnglish() ? "Before success rate" : "重试前成功率"}: ${pct(retryCmp.before_success_rate || 0)}\n- ${isEnglish() ? "After success rate" : "重试后成功率"}: ${pct(retryCmp.after_success_rate || 0)}\n- ${isEnglish() ? "Source failure before" : "原失败重试前占比"}: ${pct(retryCmp.source_failure_before_rate || 0)}\n- ${isEnglish() ? "Source failure after" : "原失败重试后占比"}: ${pct(retryCmp.source_failure_after_rate || 0)}\n`
    : "";
  const memoryLine = currentNlpResult?.experiment_memory
    ? `\n## ${isEnglish() ? "Experiment Memory" : "实验记忆"}\n\n- ${isEnglish() ? "Recorded" : "是否写入"}: ${currentNlpResult.experiment_memory.recorded ? "yes" : "no"}\n- ${isEnglish() ? "Total records" : "历史记录数"}: ${currentNlpResult.experiment_memory.summary?.total_records || 0}\n`
    : "";
  const agentConclusion = currentAgentAnalysis?.agent_conclusion
    ? `\n## ${isEnglish() ? "Agent Conclusion" : "智能体结论"}\n\n${currentAgentAnalysis.agent_conclusion}\n`
    : "";

  if (isEnglish()) {
    currentReport = `# Robot Research Experiment Agent Report

## Research Goal

${$("goalInput").value.trim()}

## Experiment Setup
${routeLine}- Number of runs: ${currentSummary.numRuns}
- Variables: ${variableNames}
- Object library scope: ${(objectLibrary || []).map(objectName).join(", ")}

## Overall Results

- Overall success rate: ${pct(currentSummary.successRate)}
- Main failure type: ${currentSummary.mainFailure[0]}
- Explainable failure share: ${pct(currentSummary.explainableRate)}

## Failure Distribution

${failureLines}

## Key Findings

${findingLines}

## Next Experiment Suggestions

${recLines}
${evalLine}${retryLine}${memoryLine}${capabilityLine ? `\n## Capability Boundary\n\n${capabilityLine}\n` : ""}${executionLine}${agentProviderLine}${agentLines}${apiCallLine}${agentConclusion}`;
  } else {
    currentReport = `# 机器人研发实验智能体报告

## 研发目标

${$("goalInput").value.trim()}

## 实验设置
${routeLine}- 实验总数: ${currentSummary.numRuns}
- 变量: ${variableNames}
- 物体模型库范围: ${(objectLibrary || []).map(objectName).join(", ")}

## 总体结果

- 总体成功率: ${pct(currentSummary.successRate)}
- 主要失败类型: ${currentSummary.mainFailure[0]}
- 可解释失败占比: ${pct(currentSummary.explainableRate)}

## 失败类型分布

${failureLines}

## 关键发现

${findingLines}

## 下一轮实验建议

${recLines}
${evalLine}${retryLine}${memoryLine}${capabilityLine ? `\n## 能力边界\n\n${capabilityLine}\n` : ""}${executionLine}${agentLines}${apiCallLine}${agentConclusion}`;
  }

  $("reportPreview").textContent = currentReport;
}

function downloadReport() {
  if (!currentReport) makeReport();
  const blob = new Blob([currentReport], { type: "text/markdown;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "robot_research_agent_report.md";
  a.click();
  URL.revokeObjectURL(url);
}

function resetDemo() {
  $("goalInput").value = defaultGoal;
  experimentSpace = {
    skill_id: ["pick_lift", "reach_touch", "contact_sweep", "tool_contact_sweep", "peg_insert"],
    object_id: ["rect_block", "cube_5cm", "cylinder_can", "cube_7cm", "screw_head", "insertion_socket"],
    tool_id: ["hammer", "spatula", "screwdriver", "peg"],
    object_position: ["center", "left", "right"],
    friction: ["low", "medium", "high"],
    grasp_height_delta: ["nominal", "low", "high"],
    sweep_scale: ["short", "nominal", "long"]
  };
  currentTask = {
    task_id: "fr3_arm_primitives",
    title: "General FR3 arm primitive experiments",
    description: "Default task: real FR3 + Franka Hand MuJoCo primitives for grasp, touch, tool contact, and peg insertion.",
    experiment_space: experimentSpace,
    execution_kind: "robot_arm_skill_simulation",
    manipulation_actor: "Franka FR3 arm with Franka Hand"
  };
  currentRuns = [];
  currentSummary = null;
  currentReport = "";
  currentTrace = null;
  currentAgentDesign = null;
  currentAgentAnalysis = null;
  currentAgentTrace = [];
  currentAgentApiCalls = null;
  currentExecutionPlan = null;
  currentAgentProvider = null;
  currentNlpResult = null;
  displayParsedTaskSpec(null);
  renderAgentPipeline(null);
  renderEvaluationPanel(null);
  renderClosedLoopPanel(null);
  renderMemoryPanel(null);
  $("runStatus").textContent = "未运行";
  if ($("nlpSampleSize")) $("nlpSampleSize").value = "27";
  if ($("robotSelect")) $("robotSelect").value = "";
  updateRobotInfo();
  $("mRuns").textContent = "--";
  $("mSuccess").textContent = "--";
  $("mFailure").textContent = "--";
  $("mExplain").textContent = "--";
  $("failureChart").innerHTML = "";
  $("variableChart").innerHTML = "";
  $("findings").innerHTML = "";
  $("recommendations").innerHTML = "";
  $("logRows").innerHTML = "";
  $("reportPreview").textContent = "请先运行 NLP 管线。";
  renderEmptyPlan();
  clearReplay();
}

function updateNlpProgress(step, detail, percent) {
  const statusEl = $("runStatus");
  const bar = $("nlpProgressBar");
  const barText = $("nlpProgressText");
  if (statusEl) statusEl.textContent = `NLP 管线 · ${step}: ${detail}`;
  if (bar) bar.style.width = `${percent}%`;
  if (barText) barText.textContent = `${step} · ${percent}%`;
}

async function pollNlpProgress() {
  const baseEndpoint = $("apiEndpoint").value.trim().replace(/\/api\/run_experiments$/, "");
  try {
    const resp = await fetch(`${baseEndpoint}/api/nlp/status`);
    if (resp.ok) {
      const data = await resp.json();
      updateNlpProgress(data.step || "...", data.detail || "", data.percent || 0);
    }
  } catch { /* ignore poll errors */ }
}

async function runNlpPipeline() {
  const goal = $("goalInput").value.trim();
  if (!goal) {
    showToast(isEnglish() ? "Please enter a research goal." : "请输入研发目标。");
    return;
  }
  if ($("nlpBtn").disabled) return;
  const limit = Number($("nlpSampleSize")?.value || $("sampleSize")?.value || 27);
  const nlpBtn = $("nlpBtn");
  nlpBtn.disabled = true;
  nlpBtn.textContent = "NLP 运行中...";
  $("reportPreview").textContent = "NLP 管线正在运行，包含真实 MuJoCo 物理仿真，请稍候...";
  const startTime = Date.now();

  // Poll progress every 500ms
  const progressTimer = setInterval(pollNlpProgress, 500);
  updateNlpProgress("初始化", "正在启动 NLP 管线...", 0);

  try {
    const payload = await fetchNlpRun(goal, limit);
    clearInterval(progressTimer);
    applyNlpPayload(payload);
    renderNlpResults(payload);
    currentTrace = payload.demo_trace || null;
    updateNlpProgress("完成", `${payload.num_runs || payload.runs?.length || 0} 组实验`, 100);
    $("runStatus").textContent = `NLP 管线完成 · ${payload.robot_id || "auto"} · ${payload.num_runs || payload.runs?.length || 0} 组`;
    makeReport();
    showToast(isEnglish()
      ? `NLP pipeline complete: ${payload.robot_id || "auto"}, ${payload.num_runs || 0} runs.`
      : `NLP 管线完成：${payload.robot_id || "自动选择"}，${payload.num_runs || 0} 组实验。`, 5000);
  } catch (error) {
    clearInterval(progressTimer);
    updateNlpProgress("失败", error.message, 0);
    $("runStatus").textContent = "NLP 管线失败";
    showToast(`NLP 管线错误：${error.message}`, 7000);
    console.error(error);
  } finally {
    nlpBtn.disabled = false;
    nlpBtn.textContent = "一键运行 NLP 管线";
  }
}

document.addEventListener("DOMContentLoaded", () => {
  $("planBtn")?.addEventListener("click", renderPlan);
  $("runBtn")?.addEventListener("click", runExperiment);
  $("nlpBtn")?.addEventListener("click", runNlpPipeline);
  $("reportBtn")?.addEventListener("click", makeReport);
  $("downloadBtn")?.addEventListener("click", downloadReport);
  $("resetBtn")?.addEventListener("click", resetDemo);
  $("outputLanguage")?.addEventListener("change", () => {
    renderTaskSelect();
    if (currentNlpResult) {
      renderPlan();
    } else {
      renderEmptyPlan();
    }
    renderAssetRegistry();
    renderRobotSelect();
    if (currentSummary) makeReport();
  });
  $("taskSelect")?.addEventListener("change", updateCustomTaskVisibility);
  $("robotSelect")?.addEventListener("change", updateRobotInfo);
  $("replayBtn")?.addEventListener("click", () => {
    if ((currentTrace?.replays?.length || currentTrace?.image_frames?.length) && currentTaskUsesRobotArm()) {
      renderReplay(currentTrace);
    }
  });
  renderEmptyPlan();
  const endpoint = $("apiEndpoint")?.value?.trim() || "http://127.0.0.1:8765/api/run_experiments";
  fetchTaskCatalog(endpoint);
  fetchObjectLibrary(endpoint);
  fetchAssetRegistry(endpoint);
  fetchRobotCatalog(endpoint);
  clearReplay();
});
