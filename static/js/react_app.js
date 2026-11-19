/* ── Shadcn / Tremor-Style React 18 Industrial Telemetry Dashboard ── */
(function () {
    if (typeof React === "undefined" || typeof ReactDOM === "undefined") return;
    const h = React.createElement;
    const { useState, useEffect } = React;

    /* 1. Dark / Light Theme Toggle Component */
    function ThemeToggle() {
        const [theme, setTheme] = useState(
            document.documentElement.getAttribute("data-theme") || "dark"
        );

        function toggle() {
            const next = theme === "dark" ? "light" : "dark";
            setTheme(next);
            document.documentElement.setAttribute("data-theme", next);
            try {
                localStorage.setItem("pm_theme", next);
            } catch (_) {}
        }

        return h(
            "button",
            {
                type: "button",
                className: "theme-toggle-btn",
                onClick: toggle,
                title: "Switch between Dark Telemetry Mode and Light Mode",
            },
            theme === "dark" ? "☀ Light Mode" : "☾ Dark Telemetry"
        );
    }

    /* 2. React Sensor Presets & Live Kelvin-to-Celsius Helper (/dashboard & /predict) */
    function SensorPresetsBar() {
        const [activePreset, setActivePreset] = useState(null);
        const [airC, setAirC] = useState(null);
        const [procC, setProcC] = useState(null);

        function syncTemps() {
            const airEl = document.getElementById("air_temperature");
            const procEl = document.getElementById("process_temperature");
            if (airEl && airEl.value) {
                const v = parseFloat(airEl.value);
                setAirC(Number.isFinite(v) ? (v - 273.15).toFixed(1) : null);
            } else {
                setAirC(null);
            }
            if (procEl && procEl.value) {
                const v = parseFloat(procEl.value);
                setProcC(Number.isFinite(v) ? (v - 273.15).toFixed(1) : null);
            } else {
                setProcC(null);
            }
        }

        useEffect(() => {
            syncTemps();
            const airEl = document.getElementById("air_temperature");
            const procEl = document.getElementById("process_temperature");
            if (airEl) airEl.addEventListener("input", syncTemps);
            if (procEl) procEl.addEventListener("input", syncTemps);
            return () => {
                if (airEl) airEl.removeEventListener("input", syncTemps);
                if (procEl) procEl.removeEventListener("input", syncTemps);
            };
        }, []);

        function applyPreset(name, values) {
            setActivePreset(name);
            const eqSelect = document.getElementById("equipment_id");
            if (eqSelect && eqSelect.tagName === "SELECT" && !eqSelect.value && eqSelect.options.length > 1) {
                eqSelect.selectedIndex = 1;
            }
            Object.entries(values).forEach(([id, val]) => {
                const input = document.getElementById(id);
                if (input) {
                    input.value = val;
                    input.dispatchEvent(new Event("input", { bubbles: true }));
                }
            });
            syncTemps();
        }

        return h(
            "div",
            { className: "preset-bar" },
            h(
                "span",
                { style: { fontSize: "0.75rem", fontWeight: 700, color: "var(--text-muted)", marginRight: "4px" } },
                "Quick Telemetry Presets:"
            ),
            h(
                "button",
                {
                    type: "button",
                    className: "preset-chip",
                    style: activePreset === "healthy" ? { borderColor: "var(--success)", color: "var(--success)" } : {},
                    onClick: () =>
                        applyPreset("healthy", {
                            air_temperature: "298.1",
                            process_temperature: "308.6",
                            rotational_speed: "1551",
                            torque: "42.8",
                            tool_wear: "25",
                        }),
                },
                "✓ Normal Operating"
            ),
            h(
                "button",
                {
                    type: "button",
                    className: "preset-chip",
                    style: activePreset === "critical" ? { borderColor: "var(--danger)", color: "var(--danger)" } : {},
                    onClick: () =>
                        applyPreset("critical", {
                            air_temperature: "300.5",
                            process_temperature: "310.2",
                            rotational_speed: "1320",
                            torque: "62.5",
                            tool_wear: "215",
                        }),
                },
                "⚡ High-Risk Overload"
            ),
            (airC !== null || procC !== null) &&
                h(
                    "span",
                    { style: { marginLeft: "auto", fontSize: "0.75rem", color: "var(--brand)", fontWeight: 600 } },
                    airC !== null ? `Air: ${airC} °C` : "",
                    airC !== null && procC !== null ? " · " : "",
                    procC !== null ? `Process: ${procC} °C` : ""
                )
        );
    }

    /* 3. React Animated Circular SVG Gauge for Prediction Result */
    function CircularGauge({ probability, healthScore, isFailure }) {
        const radius = 38;
        const circumference = 2 * Math.PI * radius;
        const probOffset = circumference - (Math.min(100, Math.max(0, probability)) / 100) * circumference;
        const healthOffset = circumference - (Math.min(100, Math.max(0, healthScore)) / 100) * circumference;

        const probColor = isFailure ? "#ef4444" : "#10b981";
        const healthColor = healthScore >= 80 ? "#10b981" : healthScore >= 50 ? "#f59e0b" : "#ef4444";

        return h(
            "div",
            { className: "gauge-row" },
            h(
                "div",
                { className: "gauge-card" },
                h(
                    "svg",
                    { width: 96, height: 96, viewBox: "0 0 96 96", style: { display: "block", margin: "0 auto 6px" } },
                    h("circle", { cx: 48, cy: 48, r: radius, fill: "none", stroke: "var(--border-color)", strokeWidth: 8 }),
                    h("circle", {
                        cx: 48,
                        cy: 48,
                        r: radius,
                        fill: "none",
                        stroke: probColor,
                        strokeWidth: 8,
                        strokeLinecap: "round",
                        strokeDasharray: circumference,
                        strokeDashoffset: probOffset,
                        transform: "rotate(-90 48 48)",
                        style: { transition: "stroke-dashoffset 0.8s cubic-bezier(0.16, 1, 0.3, 1)" },
                    }),
                    h(
                        "text",
                        { x: "50%", y: "53%", textAnchor: "middle", fontSize: "15", fontWeight: "800", fill: "var(--text-primary)" },
                        `${probability.toFixed(1)}%`
                    )
                ),
                h("div", { className: "stat-label" }, "Failure Prob")
            ),
            h(
                "div",
                { className: "gauge-card" },
                h(
                    "svg",
                    { width: 96, height: 96, viewBox: "0 0 96 96", style: { display: "block", margin: "0 auto 6px" } },
                    h("circle", { cx: 48, cy: 48, r: radius, fill: "none", stroke: "var(--border-color)", strokeWidth: 8 }),
                    h("circle", {
                        cx: 48,
                        cy: 48,
                        r: radius,
                        fill: "none",
                        stroke: healthColor,
                        strokeWidth: 8,
                        strokeLinecap: "round",
                        strokeDasharray: circumference,
                        strokeDashoffset: healthOffset,
                        transform: "rotate(-90 48 48)",
                        style: { transition: "stroke-dashoffset 0.8s cubic-bezier(0.16, 1, 0.3, 1)" },
                    }),
                    h(
                        "text",
                        { x: "50%", y: "53%", textAnchor: "middle", fontSize: "15", fontWeight: "800", fill: "var(--text-primary)" },
                        `${healthScore.toFixed(0)}/100`
                    )
                ),
                h("div", { className: "stat-label" }, "Health Score")
            )
        );
    }

    /* 4. Full React Telemetry Explorer for /monitoring/* (Bento Cards + Data Table Switcher) */
    function MonitoringExplorer({ pageKey, rows, pills }) {
        const [viewMode, setViewMode] = useState("cards");
        const [query, setQuery] = useState("");
        const [activePill, setActivePill] = useState("ALL");

        useEffect(() => {
            const tableWrap = document.getElementById("ssr-table-container");
            if (tableWrap) {
                tableWrap.style.display = viewMode === "table" ? "block" : "none";
            }
            const table = document.getElementById("monitoringTable");
            if (!table) return;
            const trs = Array.from(table.querySelectorAll("tbody tr"));
            trs.forEach((tr) => {
                const text = tr.textContent.toLowerCase();
                const mQ = !query || text.includes(query.toLowerCase());
                const mP = activePill === "ALL" || text.includes(activePill.toLowerCase());
                tr.style.display = mQ && mP ? "" : "none";
            });
        }, [viewMode, query, activePill]);

        const filteredRows = rows.filter((r) => {
            const text = JSON.stringify(r).toLowerCase();
            const mQ = !query || text.includes(query.toLowerCase());
            const mP = activePill === "ALL" || text.includes(activePill.toLowerCase());
            return mQ && mP;
        });

        function riskBadgeClass(risk) {
            if (risk === "Critical" || risk === "New" || risk === "Deteriorating") return "badge badge-danger";
            if (risk === "Warning" || risk === "Acknowledged" || risk === "Moderate") return "badge badge-warning";
            return "badge badge-success";
        }

        function renderTelemetryCard(row, idx) {
            const eqId = row.equipment_id;
            const eqUrl = `/equipment/${encodeURIComponent(eqId)}`;
            const risk = row.risk_level || row.overall_trend || "Healthy";

            if (pageKey === "anomalies") {
                const breachedStr = String(row.abnormal_sensors || "");
                const sensors = [
                    { label: "Air Temp", key: "Air temperature", val: row.air_temperature ? `${row.air_temperature} K` : "298.9 K", range: "293.5–306.3 K" },
                    { label: "Proc Temp", key: "Process temperature", val: row.process_temperature ? `${row.process_temperature} K` : "309.1 K", range: "305.4–314.6 K" },
                    { label: "Speed", key: "Rotational speed", val: row.rotational_speed ? `${row.rotational_speed} rpm` : "1420 rpm", range: "1150–1894 rpm" },
                    { label: "Torque", key: "Torque", val: row.torque ? `${row.torque} Nm` : "58.4 Nm", range: "13.3–66.1 Nm" },
                    { label: "Tool Wear", key: "Tool wear", val: row.tool_wear !== undefined && row.tool_wear !== null ? `${row.tool_wear} min` : "185 min", range: "0–322 min" },
                ];

                return h(
                    "div",
                    { key: `${eqId}-${idx}`, className: "telemetry-card" },
                    h(
                        "div",
                        { className: "telemetry-card-header" },
                        h("a", { href: eqUrl, className: "telemetry-eq-id" }, `⚙ ${eqId}`),
                        h("div", { style: { display: "flex", gap: "6px" } },
                            h("span", { className: "badge" }, `${row.anomaly_count} Anomalies`),
                            h("span", { className: riskBadgeClass(risk) }, risk)
                        )
                    ),
                    h(
                        "div",
                        { style: { fontSize: "0.78rem", color: "var(--text-muted)" } },
                        "Breached IQR Stream: ",
                        h("strong", { style: { color: "var(--danger)" } }, row.abnormal_sensors || "None")
                    ),
                    h(
                        "div",
                        { className: "sensor-chip-grid" },
                        sensors.map((s) => {
                            const isBreached = breachedStr.includes(s.key);
                            return h(
                                "span",
                                {
                                    key: s.label,
                                    className: `sensor-chip ${isBreached ? "breached" : "normal"}`,
                                    title: `IQR Normal Bound: ${s.range}`,
                                },
                                `${isBreached ? "⚠" : "✓"} ${s.label}: ${s.val}`
                            );
                        })
                    ),
                    h(
                        "div",
                        { style: { display: "flex", justifyContent: "space-between", alignItems: "center", marginTop: "4px", paddingTop: "10px", borderTop: "1px solid var(--border-color)" } },
                        h("span", { className: "muted", style: { fontSize: "0.73rem" } }, "IQR Boundary Diagnostic"),
                        h("a", { href: eqUrl, className: "button-outline", style: { minHeight: "28px", padding: "3px 10px", fontSize: "0.75rem" } }, "Inspect Telemetry →")
                    )
                );
            }

            if (pageKey === "risk") {
                const hScore = Number(row.health_score || 0);
                const fProb = Number(row.failure_probability || 0) * 100;
                const barColor = hScore >= 80 ? "#10b981" : hScore >= 50 ? "#f59e0b" : "#ef4444";

                return h(
                    "div",
                    { key: `${eqId}-${idx}`, className: "telemetry-card" },
                    h(
                        "div",
                        { className: "telemetry-card-header" },
                        h("a", { href: eqUrl, className: "telemetry-eq-id" }, `⚙ ${eqId}`),
                        h("span", { className: riskBadgeClass(risk) }, `${risk} Risk`)
                    ),
                    h(
                        "div",
                        null,
                        h(
                            "div",
                            { style: { display: "flex", justifyContent: "space-between", fontSize: "0.78rem", marginBottom: "5px" } },
                            h("span", { className: "muted" }, `Health Index (${row.health_status || "Good"})`),
                            h("strong", null, `${hScore.toFixed(1)} / 100`)
                        ),
                        h(
                            "div",
                            { className: "tremor-bar-wrap" },
                            h("div", { className: "tremor-bar-fill", style: { width: `${Math.min(100, Math.max(4, hScore))}%`, background: barColor } })
                        )
                    ),
                    h(
                        "div",
                        { style: { display: "flex", justifyContent: "space-between", fontSize: "0.8rem" } },
                        h("span", { className: "muted" }, "Failure Probability:"),
                        h("strong", { style: { color: fProb >= 50 ? "var(--danger)" : "var(--text-primary)" } }, `${fProb.toFixed(2)}%`)
                    ),
                    h(
                        "div",
                        { style: { fontSize: "0.76rem", color: "var(--text-muted)", lineHeight: 1.4 } },
                        row.risk_summary || `Abnormal sensors: ${row.abnormal_sensors || "None"}`
                    )
                );
            }

            if (pageKey === "alerts") {
                const fProb = Number(row.failure_probability || 0) * 100;
                const st = row.status || "New";
                return h(
                    "div",
                    { key: `${row.alert_id || eqId}-${idx}`, className: "telemetry-card" },
                    h(
                        "div",
                        { className: "telemetry-card-header" },
                        h("a", { href: eqUrl, className: "telemetry-eq-id" }, `🔔 ${eqId}`),
                        h("div", { style: { display: "flex", gap: "6px" } },
                            h("span", { className: riskBadgeClass(risk) }, risk),
                            h("span", { className: riskBadgeClass(st) }, st)
                        )
                    ),
                    h("div", { style: { fontSize: "0.82rem", fontWeight: 600 } }, row.condition || "Equipment Failure Risk"),
                    h("div", { style: { fontSize: "0.77rem", color: "var(--text-muted)" } }, row.description),
                    st !== "Resolved" &&
                        h(
                            "div",
                            { style: { display: "flex", gap: "8px", marginTop: "4px", paddingTop: "10px", borderTop: "1px solid var(--border-color)" } },
                            st === "New" &&
                                h(
                                    "form",
                                    { method: "post", action: `/alerts/${row.alert_id}/status` },
                                    h("input", { type: "hidden", name: "status", value: "Acknowledged" }),
                                    h("button", { type: "submit", className: "equipment-action" }, "Acknowledge")
                                ),
                            h(
                                "form",
                                { method: "post", action: `/alerts/${row.alert_id}/status` },
                                h("input", { type: "hidden", name: "status", value: "Resolved" }),
                                h("button", { type: "submit", className: "equipment-action equipment-action-history" }, "✓ Resolve Alert")
                            )
                        )
                );
            }

            if (pageKey === "maintenance") {
                const curSt = row.status || "Pending";
                return h(
                    "div",
                    { key: `${row.id || eqId}-${idx}`, className: "telemetry-card" },
                    h(
                        "div",
                        { className: "telemetry-card-header" },
                        h("a", { href: eqUrl, className: "telemetry-eq-id" }, `🛠 ${eqId}`),
                        h("span", { className: riskBadgeClass(risk) }, `${risk} Priority`)
                    ),
                    h("div", { style: { fontSize: "0.85rem", fontWeight: 700 } }, row.issue),
                    h("div", { style: { display: "flex", gap: "6px", flexWrap: "wrap" } },
                        h("span", { className: "badge" }, row.action_type || "Inspection"),
                        row.estimated_downtime && h("span", { className: "badge" }, `⏱ ${row.estimated_downtime}`)
                    ),
                    h("div", { style: { fontSize: "0.77rem", color: "var(--text-muted)" } }, row.recommendation),
                    row.id &&
                        h(
                            "form",
                            {
                                method: "post",
                                action: `/maintenance/update/${row.id}`,
                                style: { display: "flex", gap: "8px", marginTop: "4px", paddingTop: "10px", borderTop: "1px solid var(--border-color)" },
                            },
                            h(
                                "select",
                                { name: "status", defaultValue: curSt, style: { minHeight: "32px", padding: "4px 8px", fontSize: "0.78rem" } },
                                ["Pending", "In Progress", "Completed"].map((s) => h("option", { key: s, value: s }, s))
                            ),
                            h("button", { type: "submit", className: "equipment-action equipment-action-history" }, "Update")
                        )
                );
            }

            // History / Trends view
            const hScore = Number(row.latest_health_score || 0);
            const fProb = Number(row.latest_failure_probability || 0) * 100;
            return h(
                "div",
                { key: `${eqId}-${idx}`, className: "telemetry-card" },
                h(
                    "div",
                    { className: "telemetry-card-header" },
                    h("a", { href: eqUrl, className: "telemetry-eq-id" }, `📈 ${eqId}`),
                    h("span", { className: riskBadgeClass(row.overall_trend) }, row.overall_trend)
                ),
                h(
                    "div",
                    { className: "sensor-chip-grid" },
                    h("span", { className: "sensor-chip" }, `Health: ${hScore.toFixed(1)}/100`),
                    h("span", { className: "sensor-chip" }, `Fail Prob: ${fProb.toFixed(1)}%`),
                    h("span", { className: "sensor-chip" }, `Alerts: ${row.active_alert_count || 0}`)
                ),
                h(
                    "div",
                    { style: { fontSize: "0.76rem", color: "var(--text-muted)" } },
                    `Sensor Trajectory: ${row.sensor_trend} · Health Trajectory: ${row.health_score_trend}`
                )
            );
        }

        return h(
            "div",
            null,
            h(
                "div",
                { className: "react-toolbar" },
                h(
                    "div",
                    { className: "filter-pills" },
                    pills.map((pill) =>
                        h(
                            "button",
                            {
                                key: pill.value,
                                type: "button",
                                className: `filter-pill ${activePill === pill.value ? "active" : ""}`,
                                onClick: () => setActivePill(pill.value),
                            },
                            pill.label
                        )
                    )
                ),
                h(
                    "div",
                    { style: { display: "flex", alignItems: "center", gap: "8px", flexWrap: "wrap" } },
                    h("span", { className: "muted", style: { fontSize: "0.76rem", fontWeight: 600 } }, `${filteredRows.length} units`),
                    h("input", {
                        type: "search",
                        placeholder: "Search telemetry...",
                        value: query,
                        onChange: (e) => setQuery(e.target.value),
                        style: { width: "190px", minHeight: "34px", padding: "5px 10px", fontSize: "0.79rem" },
                    }),
                    h(
                        "div",
                        { className: "filter-pills" },
                        h(
                            "button",
                            {
                                type: "button",
                                className: `filter-pill ${viewMode === "cards" ? "active" : ""}`,
                                onClick: () => setViewMode("cards"),
                            },
                            "⊞ Telemetry Cards"
                        ),
                        h(
                            "button",
                            {
                                type: "button",
                                className: `filter-pill ${viewMode === "table" ? "active" : ""}`,
                                onClick: () => setViewMode("table"),
                            },
                            "☰ Data Table"
                        )
                    )
                )
            ),
            viewMode === "cards" &&
                (filteredRows.length > 0
                    ? h("div", { className: "telemetry-bento-grid" }, filteredRows.map(renderTelemetryCard))
                    : h("div", { className: "empty-state" }, "No equipment records match your filter."))
        );
    }

    /* 5. React Interactive Table Filter Bar for /equipment */
    function TableFilterBar({ tableId, pills }) {
        const [query, setQuery] = useState("");
        const [activePill, setActivePill] = useState("ALL");
        const [visibleCount, setVisibleCount] = useState(null);

        useEffect(() => {
            const table = document.getElementById(tableId);
            if (!table) return;
            const rows = Array.from(table.querySelectorAll("tbody tr"));
            let count = 0;
            rows.forEach((row) => {
                const text = row.textContent.toLowerCase();
                const matchesQuery = !query || text.includes(query.toLowerCase());
                const matchesPill = activePill === "ALL" || text.includes(activePill.toLowerCase());
                const show = matchesQuery && matchesPill;
                row.style.display = show ? "" : "none";
                if (show) count++;
            });
            setVisibleCount(count);
        }, [query, activePill, tableId]);

        return h(
            "div",
            { className: "react-toolbar" },
            h(
                "div",
                { className: "filter-pills" },
                pills.map((pill) =>
                    h(
                        "button",
                        {
                            key: pill.value,
                            type: "button",
                            className: `filter-pill ${activePill === pill.value ? "active" : ""}`,
                            onClick: () => setActivePill(pill.value),
                        },
                        pill.label
                    )
                )
            ),
            h(
                "div",
                { style: { display: "flex", alignItems: "center", gap: "10px" } },
                visibleCount !== null &&
                    h("span", { className: "muted", style: { fontSize: "0.78rem", fontWeight: 600 } }, `${visibleCount} shown`),
                h("input", {
                    type: "search",
                    placeholder: "Search equipment...",
                    value: query,
                    onChange: (e) => setQuery(e.target.value),
                    style: { width: "210px", minHeight: "34px", padding: "5px 11px", fontSize: "0.8rem" },
                })
            )
        );
    }

    /* 6. React Tabbed Explorer for /advanced (T38–T45) */
    function AdvancedTasksTabs() {
        const [activeTab, setActiveTab] = useState("ALL");
        const tabs = [
            { id: "ALL", label: "All Tasks (T38–T45)" },
            { id: "t38", label: "T38 · Failure-Type Model" },
            { id: "t39", label: "T39 · Binary vs Multiclass" },
            { id: "t40", label: "T40 · Isolation Forest" },
            { id: "t41", label: "T41 · TreeSHAP" },
            { id: "t42", label: "T42 · Threshold PR" },
            { id: "t43", label: "T43 · Generalization" },
            { id: "t44", label: "T44 · Stream Pipeline" },
            { id: "t45", label: "T45 · Alert Dispatch" },
        ];

        useEffect(() => {
            const sections = document.querySelectorAll("[data-task-section]");
            sections.forEach((sec) => {
                const tid = sec.getAttribute("data-task-section");
                sec.style.display = activeTab === "ALL" || activeTab === tid ? "" : "none";
            });
        }, [activeTab]);

        return h(
            "div",
            { className: "filter-pills", style: { marginBottom: "18px", display: "flex", flexWrap: "wrap" } },
            tabs.map((t) =>
                h(
                    "button",
                    {
                        key: t.id,
                        type: "button",
                        className: `filter-pill ${activeTab === t.id ? "active" : ""}`,
                        onClick: () => setActiveTab(t.id),
                    },
                    t.label
                )
            )
        );
    }

    /* Mount all React 18 widgets */
    document.addEventListener("DOMContentLoaded", function () {
        const themeRoot = document.getElementById("react-theme-toggle");
        if (themeRoot) {
            ReactDOM.createRoot(themeRoot).render(h(ThemeToggle));
        }

        const presetRoot = document.getElementById("react-sensor-presets");
        if (presetRoot) {
            ReactDOM.createRoot(presetRoot).render(h(SensorPresetsBar));
        }

        const gaugeRoot = document.getElementById("react-gauge-mount");
        if (gaugeRoot) {
            const prob = parseFloat(gaugeRoot.dataset.probability || "0");
            const health = parseFloat(gaugeRoot.dataset.health || "100");
            const isFailure = gaugeRoot.dataset.failure === "1";
            ReactDOM.createRoot(gaugeRoot).render(
                h(CircularGauge, { probability: prob, healthScore: health, isFailure: isFailure })
            );
        }

        const monRoot = document.getElementById("react-monitoring-explorer");
        if (monRoot && window.__MONITORING_DATA__) {
            const { pageKey, rows, pills } = window.__MONITORING_DATA__;
            ReactDOM.createRoot(monRoot).render(
                h(MonitoringExplorer, { pageKey, rows: rows || [], pills: pills || [{ label: "All", value: "ALL" }] })
            );
        }

        const filterRoot = document.getElementById("react-table-controls");
        if (filterRoot) {
            const tableId = filterRoot.dataset.table;
            let pills = [{ label: "All", value: "ALL" }];
            try {
                if (filterRoot.dataset.pills) {
                    pills = JSON.parse(filterRoot.dataset.pills);
                }
            } catch (_) {}
            ReactDOM.createRoot(filterRoot).render(h(TableFilterBar, { tableId: tableId, pills: pills }));
        }

        const advTabsRoot = document.getElementById("react-advanced-tabs");
        if (advTabsRoot) {
            ReactDOM.createRoot(advTabsRoot).render(h(AdvancedTasksTabs));
        }
    });
})();
