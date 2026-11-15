/* ── Interactive React 18 Components for Predictive Maintenance UI ── */
(function () {
    if (typeof React === "undefined" || typeof ReactDOM === "undefined") return;
    const h = React.createElement;
    const { useState, useEffect } = React;

    /* 1. React Sensor Presets & Live Kelvin-to-Celsius Helper (/dashboard & /predict) */
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
                { style: { fontSize: "0.76rem", fontWeight: 700, color: "#64748b", marginRight: "4px" } },
                "Quick Demo Presets:"
            ),
            h(
                "button",
                {
                    type: "button",
                    className: "preset-chip",
                    style: activePreset === "healthy" ? { borderColor: "#059669", background: "#ecfdf5", color: "#059669" } : {},
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
                    style: activePreset === "critical" ? { borderColor: "#e11d48", background: "#fff1f2", color: "#e11d48" } : {},
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
                    { style: { marginLeft: "auto", fontSize: "0.75rem", color: "#4f46e5", fontWeight: 600 } },
                    airC !== null ? `Air: ${airC} °C` : "",
                    airC !== null && procC !== null ? " · " : "",
                    procC !== null ? `Process: ${procC} °C` : ""
                )
        );
    }

    /* 2. React Animated Circular SVG Gauge for Prediction Result */
    function CircularGauge({ probability, healthScore, isFailure }) {
        const radius = 38;
        const circumference = 2 * Math.PI * radius;
        const probOffset = circumference - (Math.min(100, Math.max(0, probability)) / 100) * circumference;
        const healthOffset = circumference - (Math.min(100, Math.max(0, healthScore)) / 100) * circumference;

        const probColor = isFailure ? "#e11d48" : "#059669";
        const healthColor = healthScore >= 80 ? "#059669" : healthScore >= 50 ? "#d97706" : "#e11d48";

        return h(
            "div",
            { className: "gauge-row" },
            h(
                "div",
                { className: "gauge-card" },
                h(
                    "svg",
                    { width: 96, height: 96, viewBox: "0 0 96 96", style: { display: "block", margin: "0 auto 6px" } },
                    h("circle", { cx: 48, cy: 48, r: radius, fill: "none", stroke: "#e2e8f0", strokeWidth: 8 }),
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
                        { x: "50%", y: "53%", textAnchor: "middle", fontSize: "15", fontWeight: "800", fill: "#0f172a" },
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
                    h("circle", { cx: 48, cy: 48, r: radius, fill: "none", stroke: "#e2e8f0", strokeWidth: 8 }),
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
                        { x: "50%", y: "53%", textAnchor: "middle", fontSize: "15", fontWeight: "800", fill: "#0f172a" },
                        `${healthScore.toFixed(0)}/100`
                    )
                ),
                h("div", { className: "stat-label" }, "Health Score")
            )
        );
    }

    /* 3. React Interactive Table Filter & Pill Bar (/equipment & /monitoring/*) */
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
                    placeholder: "Quick search...",
                    value: query,
                    onChange: (e) => setQuery(e.target.value),
                    style: { width: "210px", minHeight: "34px", padding: "5px 11px", fontSize: "0.8rem" },
                })
            )
        );
    }

    /* Mount React components where containers exist */
    document.addEventListener("DOMContentLoaded", function () {
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
    });
})();
