const el = (id) => document.getElementById(id);

const refs = {
  kpiSuppliers: el("kpiSuppliers"),
  kpiInventory: el("kpiInventory"),
  kpiLowStock: el("kpiLowStock"),
  kpiOrders: el("kpiOrders"),
  kpiSpend: el("kpiSpend"),
  inventoryTable: el("inventoryTable"),
  supplierTable: el("supplierTable"),
  ordersTable: el("ordersTable"),
  alertList: el("alertList"),
  forecastResult: el("forecastResult"),
  toast: el("toast"),
};

const forms = {
  inventory: el("inventoryForm"),
  supplier: el("supplierForm"),
  order: el("orderForm"),
  forecast: el("forecastForm"),
};

const buttons = {
  refreshDashboard: el("refreshDashboard"),
  loadInventory: el("loadInventory"),
  loadSuppliers: el("loadSuppliers"),
  loadOrders: el("loadOrders"),
};

function showToast(message, timeout = 2200) {
  refs.toast.textContent = message;
  refs.toast.classList.remove("hidden");
  setTimeout(() => refs.toast.classList.add("hidden"), timeout);
}

async function api(path, options = {}) {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });

  const json = await res.json().catch(() => ({}));
  if (!res.ok) {
    const detail = json.detail ? JSON.stringify(json.detail) : `HTTP ${res.status}`;
    throw new Error(detail);
  }
  return json;
}

function statusTag(stock, reorder) {
  if (stock < reorder * 0.6) return '<span class="tag danger">Critical</span>';
  if (stock <= reorder) return '<span class="tag warning">Reorder</span>';
  return '<span class="tag ok">Healthy</span>';
}

function money(v) {
  return `UGX ${Number(v || 0).toLocaleString()}`;
}

async function loadDashboard() {
  const data = await api("/api/dashboard");
  refs.kpiSuppliers.textContent = data.kpis.total_suppliers;
  refs.kpiInventory.textContent = data.kpis.total_inventory_items;
  refs.kpiLowStock.textContent = data.kpis.low_stock_count;
  refs.kpiOrders.textContent = data.kpis.active_orders;
  refs.kpiSpend.textContent = money(data.kpis.monthly_purchase_cost);

  refs.alertList.innerHTML = "";
  if (!data.low_stock_items.length) {
    refs.alertList.innerHTML = "<li style='color: var(--ok); font-weight:600'>✅ You're good! All items are well-stocked.</li>";
  } else {
    data.low_stock_items.forEach((item) => {
      const li = document.createElement("li");
      const critical = item.current_stock < item.reorder_level * 0.6;
      const icon = critical ? "🚨" : "⚠️";
      li.innerHTML = `${icon} <strong>${item.name}</strong>: ${item.current_stock} ${item.unit} left (should have ${item.reorder_level})`;
      refs.alertList.appendChild(li);
    });
  }
}

async function loadInventory() {
  const items = await api("/api/inventory");
  refs.inventoryTable.innerHTML = items
    .map(
      (item) => `
        <tr>
          <td>${item.id}</td>
          <td>${item.name}</td>
          <td>${item.current_stock} ${item.unit}</td>
          <td>${item.reorder_level}</td>
          <td>${money(item.unit_cost)}</td>
          <td>${item.supplier_id || "-"}</td>
          <td>${statusTag(item.current_stock, item.reorder_level)}</td>
        </tr>
      `
    )
    .join("");
}

async function loadSuppliers() {
  const items = await api("/api/suppliers");
  refs.supplierTable.innerHTML = items
    .map(
      (s) => `
        <tr>
          <td>${s.id}</td>
          <td>${s.name}</td>
          <td>${s.contact}</td>
          <td>${s.location}</td>
          <td>${s.lead_time_days} days</td>
          <td>${s.reliability_score.toFixed(2)}</td>
        </tr>
      `
    )
    .join("");
}

function renderOrderAction(order) {
  if (order.status === "received" || order.status === "cancelled") return "-";
  return `<button class="btn" data-receive-order="${order.id}">Mark Received</button>`;
}

async function loadOrders() {
  const items = await api("/api/orders");
  refs.ordersTable.innerHTML = items
    .map(
      (o) => `
        <tr>
          <td>${o.id}</td>
          <td>${o.supplier_id}</td>
          <td>${o.expected_delivery}</td>
          <td>${o.status}</td>
          <td>${money(o.total_cost)}</td>
          <td>${renderOrderAction(o)}</td>
        </tr>
      `
    )
    .join("");

  refs.ordersTable.querySelectorAll("button[data-receive-order]").forEach((btn) => {
    btn.addEventListener("click", async () => {
      const orderId = Number(btn.dataset.receiveOrder);
      await api(`/api/orders/${orderId}/status`, {
        method: "PATCH",
        body: JSON.stringify({ status: "received" }),
      });
      showToast(`✅ Order ${orderId} received and stock updated`);
      await refreshAll();
    });
  });
}

function parseForm(form) {
  const out = Object.fromEntries(new FormData(form).entries());
  Object.keys(out).forEach((k) => {
    if (out[k] === "") {
      out[k] = null;
      return;
    }
    const n = Number(out[k]);
    if (!Number.isNaN(n)) out[k] = n;
  });
  return out;
}

forms.inventory.addEventListener("submit", async (e) => {
  e.preventDefault();
  try {
    const body = parseForm(forms.inventory);
    await api("/api/inventory", { method: "POST", body: JSON.stringify(body) });
    forms.inventory.reset();
    showToast("✅ Got it! Item added to stock");
    await refreshAll();
  } catch (err) {
    showToast(`⚠️ Couldn't add item: ${err.message}`, 3500);
  }
});

forms.supplier.addEventListener("submit", async (e) => {
  e.preventDefault();
  try {
    const body = parseForm(forms.supplier);
    await api("/api/suppliers", { method: "POST", body: JSON.stringify(body) });
    forms.supplier.reset();
    showToast("✅ Supplier saved");
    await refreshAll();
  } catch (err) {
    showToast(`⚠️ Oops: ${err.message}`, 3500);
  }
});

forms.order.addEventListener("submit", async (e) => {
  e.preventDefault();
  try {
    const body = parseForm(forms.order);
    body.items = [{ inventory_id: body.inventory_id, quantity: body.quantity }];
    delete body.inventory_id;
    delete body.quantity;

    await api("/api/orders", { method: "POST", body: JSON.stringify(body) });
    forms.order.reset();
    showToast("✅ Order placed!");
    await refreshAll();
  } catch (err) {
    showToast(`⚠️ Couldn't create order: ${err.message}`, 3500);
  }
});

forms.forecast.addEventListener("submit", async (e) => {
  e.preventDefault();
  try {
    const body = parseForm(forms.forecast);
    const data = await api("/api/forecast-procurement", {
      method: "POST",
      body: JSON.stringify(body),
    });

    const p = data.prediction;
    const recs = data.suggested_procurement.length
      ? data.suggested_procurement
          .map((x) => `<li>📦 <strong>${x.name}</strong>: grab about ${x.recommended_quantity} units (roughly ${money(x.estimated_cost)})</li>`)
          .join("")
      : "<li>✅ You're all stocked up! No top-ups needed right now.</li>";

    refs.forecastResult.classList.remove("muted");
    refs.forecastResult.innerHTML = `
      <div style="line-height: 1.8">
        <p><strong>🎯 Expected Demand:</strong> ${p.cluster_name}</p>
        <p><strong>📊 Confidence:</strong> ${(Number(p.membership_strength) * 100).toFixed(0)}% sure</p>
        <p><strong>💡 What to do:</strong> ${p.supply_action}</p>
        <p><strong>📋 Suggested Restocking:</strong></p>
        <ul style="margin:8px 0">${recs}</ul>
      </div>
    `;
  } catch (err) {
    refs.forecastResult.classList.remove("muted");
    refs.forecastResult.innerHTML = `<p style="color:#b91c1c">❌ Forecast failed: ${err.message}</p>`;
  }
});

buttons.refreshDashboard.addEventListener("click", () => refreshAll());
buttons.loadInventory.addEventListener("click", () => loadInventory());
buttons.loadSuppliers.addEventListener("click", () => loadSuppliers());
buttons.loadOrders.addEventListener("click", () => loadOrders());

async function refreshAll() {
  await Promise.all([loadDashboard(), loadInventory(), loadSuppliers(), loadOrders()]);
}

refreshAll().catch((err) => showToast(`⚠️ Couldn't load the dashboard: ${err.message}`, 4000));