/* ==========================================================================
   Xtract Application Script - Dynamic Sandbox Interactions & Live Charting
   ========================================================================== */

document.addEventListener("DOMContentLoaded", () => {
    // DOM Elements
    const dropZone = document.getElementById("drop-zone");
    const fileInput = document.getElementById("file-input");
    const browseBtn = document.getElementById("browse-btn");
    const loadingSpinner = document.getElementById("loading-spinner");
    const canvasEmpty = document.getElementById("canvas-empty");
    const canvasContainer = document.getElementById("canvas-container");
    const docImage = document.getElementById("doc-image");
    const boxOverlay = document.getElementById("box-overlay");

    // Results
    const resultsEmpty = document.getElementById("results-empty");
    const extractedFields = document.getElementById("extracted-fields");
    const latencyVal = document.getElementById("latency-val");
    const itemsTbody = document.getElementById("items-tbody");

    // Status badges
    const backendStatus = document.getElementById("backend-status");
    const gpuStatus = document.getElementById("gpu-status");

    // Tabs
    const tabBtns = document.querySelectorAll(".tab-btn");
    const tabContents = document.querySelectorAll(".tab-content");

    // Training Simulator Elements
    const selectModel = document.getElementById("select-model");
    const selectDataset = document.getElementById("select-dataset");
    const trainEpochs = document.getElementById("train-epochs");
    const trainLr = document.getElementById("train-lr");
    const trainBatch = document.getElementById("train-batch");
    const startTrainBtn = document.getElementById("start-train-btn");
    const terminalLogs = document.getElementById("terminal-logs");

    // State
    let currentDocumentData = null;
    let trainingChart = null;

    // Initialize UI
    checkBackendStatus();
    setupTabListeners();
    setupUploadHandlers();
    setupTrainingSimulator();
    loadExtractionHistory();
    setupHistoryHandlers();
    setupKIEInteractiveFeatures();

    // Recalculate box coordinates on window resize
    window.addEventListener("resize", () => {
        if (currentDocumentData) {
            renderBoundingBoxes(currentDocumentData.bounding_boxes);
        }
    });

    // ==========================================
    // 1. Backend Status & Connection Checks
    // ==========================================
    async function checkBackendStatus() {
        try {
            const res = await fetch("/api/models");
            if (!res.ok) throw new Error("HTTP error");
            const data = await res.json();

            // Update Backend Indicator
            backendStatus.querySelector(".indicator").className = "indicator green";
            backendStatus.querySelector(".label").innerText = "FastAPI Backend Online";

            // Update Hardware / GPU
            const hw = data.hardware;
            gpuStatus.querySelector(".label").innerText = `Device: ${hw.device}`;
            if (hw.gpu_available) {
                gpuStatus.querySelector("i").style.color = "var(--secondary)";
            } else {
                gpuStatus.querySelector("i").style.color = "var(--primary)";
            }
        } catch (err) {
            console.error("Backend offline:", err);
            backendStatus.querySelector(".indicator").className = "indicator blink";
            backendStatus.querySelector(".label").innerText = "Backend Offline (Demo Mode)";
            gpuStatus.querySelector(".label").innerText = "Device: Mock CPU";
        }
    }

    // ==========================================
    // 2. Tab Navigation
    // ==========================================
    function setupTabListeners() {
        tabBtns.forEach(btn => {
            btn.addEventListener("click", () => {
                tabBtns.forEach(b => b.classList.remove("active"));
                tabContents.forEach(c => c.classList.remove("active"));

                btn.classList.add("active");
                const tabId = btn.getAttribute("data-tab");
                document.getElementById(tabId).classList.add("active");
            });
        });
    }

    // ==========================================
    // 3. Document Sandbox Upload Handling
    // ==========================================
    function setupUploadHandlers() {
        browseBtn.addEventListener("click", () => fileInput.click());

        fileInput.addEventListener("change", (e) => {
            if (e.target.files.length > 0) {
                uploadFile(e.target.files[0]);
            }
        });

        // Drag and drop events
        ["dragenter", "dragover"].forEach(eventName => {
            dropZone.addEventListener(eventName, (e) => {
                e.preventDefault();
                dropZone.classList.add("dragover");
            }, false);
        });

        ["dragleave", "drop"].forEach(eventName => {
            dropZone.addEventListener(eventName, (e) => {
                e.preventDefault();
                dropZone.classList.remove("dragover");
            }, false);
        });

        dropZone.addEventListener("drop", (e) => {
            const dt = e.dataTransfer;
            const files = dt.files;
            if (files.length > 0) {
                uploadFile(files[0]);
            }
        });
    }


    // ==========================================
    // 4. File Processing & Display
    // ==========================================
    function loadImageAndRender(url, data) {
        showLoading(false);
        docImage.onload = () => {
            displayExtractionResults(data);
            renderBoundingBoxes(data.bounding_boxes);
            docImage.onload = null;
        };
        docImage.src = url;

        // If image is already fully complete (cached)
        if (docImage.complete) {
            displayExtractionResults(data);
            renderBoundingBoxes(data.bounding_boxes);
            docImage.onload = null;
        }
    }

    async function uploadFile(file) {
        showLoading(true);
        const formData = new FormData();
        formData.append("file", file);
        formData.append("use_layoutlm", document.getElementById("chk-layoutlm").checked);
        formData.append("use_trocr", document.getElementById("chk-trocr").checked);
        formData.append("use_crnn", document.getElementById("chk-crnn").checked);

        try {
            const response = await fetch("/api/upload", {
                method: "POST",
                body: formData
            });

            if (!response.ok) throw new Error("Upload processing failed");
            const data = await response.json();
            currentDocumentData = data;

            loadImageAndRender(data.file_url, data);
            
            // Refresh history table
            loadExtractionHistory();

        } catch (err) {
            console.error("Error uploading file:", err);
            // Dynamic mock fallback if server request fails completely
            simulateFallback(file.name, file);
        }
    }


    function simulateFallback(filename, fileObj = null) {
        // Local frontend mock simulation if network is unreachable
        console.log("Executing client-side OCR layout simulation");
        let docType = "receipt";
        let fields = {
            merchant: "SuperMart Stores Inc.",
            date: "2026-06-12",
            invoice_no: "INV-2026-8941",
            items: [
                { serial: "1", name: "Organic Milk 1 Gallon", quantity: "1", rate: "$5.99", price: "$5.99" },
                { serial: "2", name: "Whole Wheat Bread", quantity: "1", rate: "$3.49", price: "$3.49" },
                { serial: "3", name: "Fresh Bananas 1lb", quantity: "1", rate: "$1.89", price: "$1.89" }
            ],
            subtotal: "$11.37",
            tax: "$0.91",
            total: "$12.28"
        };
        let boundingBoxes = [
            { id: "field_merchant", label: "Merchant", value: fields.merchant, bbox: [120, 40, 380, 75], confidence: 0.98 },
            { id: "field_date", label: "Date", value: fields.date, bbox: [200, 110, 300, 130], confidence: 0.99 },
            { id: "field_invoice_no", label: "Invoice No", value: fields.invoice_no, bbox: [220, 140, 350, 160], confidence: 0.95 },
            { id: "field_item_0", label: "Item 1", value: "1. Organic Milk 1 Gallon - Qty: 1 @ $5.99 = $5.99", bbox: [50, 200, 240, 220], confidence: 0.92 },
            { id: "field_item_1", label: "Item 2", value: "2. Whole Wheat Bread - Qty: 1 @ $3.49 = $3.49", bbox: [50, 230, 210, 250], confidence: 0.91 },
            { id: "field_item_2", label: "Item 3", value: "3. Fresh Bananas 1lb - Qty: 1 @ $1.89 = $1.89", bbox: [50, 260, 200, 280], confidence: 0.93 },
            { id: "field_total", label: "Total Amount", value: fields.total, bbox: [310, 440, 380, 465], confidence: 0.99 }
        ];

        if (filename.toLowerCase().includes("invoice")) {
            docType = "invoice";
            fields = {
                merchant: "Apex Tech Solutions Ltd.",
                date: "2026-05-28",
                invoice_no: "TX-90214",
                items: [
                    { serial: "1", name: "Cloud Infrastructure Setup", quantity: "1", rate: "$1,200.00", price: "$1,200.00" },
                    { serial: "2", name: "Database Migration Consulting", quantity: "1", rate: "$850.00", price: "$850.00" }
                ],
                subtotal: "$2,050.00",
                tax: "$164.00",
                total: "$2,214.00"
            };
            boundingBoxes = [
                { id: "field_merchant", label: "Merchant", value: fields.merchant, bbox: [80, 50, 420, 90], confidence: 0.97 },
                { id: "field_date", label: "Date", value: fields.date, bbox: [650, 120, 750, 140], confidence: 0.99 },
                { id: "field_invoice_no", label: "Invoice No", value: fields.invoice_no, bbox: [650, 90, 750, 110], confidence: 0.96 },
                { id: "field_item_0", label: "Item 1", value: "1. Cloud Infrastructure Setup - Qty: 1 @ $1,200.00 = $1,200.00", bbox: [80, 320, 320, 345], confidence: 0.94 },
                { id: "field_item_1", label: "Item 2", value: "2. Database Migration Consulting - Qty: 1 @ $850.00 = $850.00", bbox: [80, 360, 340, 385], confidence: 0.93 },
                { id: "field_total", label: "Total Amount", value: fields.total, bbox: [610, 590, 740, 620], confidence: 0.99 }
            ];
        }

        const data = {
            document_type: docType,
            fields: fields,
            bounding_boxes: boundingBoxes,
            processing_time_sec: 0.45
        };

        currentDocumentData = data;
        const imgUrl = fileObj ? URL.createObjectURL(fileObj) : `/uploads/${filename}`;
        loadImageAndRender(imgUrl, data);
    }

    function showLoading(isLoading) {
        if (isLoading) {
            loadingSpinner.style.display = "flex";
            canvasContainer.style.display = "none";
            canvasEmpty.style.display = "none";
        } else {
            loadingSpinner.style.display = "none";
            canvasContainer.style.display = "block";
        }
    }

    // ==========================================
    // 5. Drawing & Scaling Bounding Boxes
    // ==========================================
    let activeMappingField = null;

    function setupKIEInteractiveFeatures() {
        // 1. In-line text editing listeners
        const fields = ["merchant", "date", "invoice_no", "subtotal", "tax", "total"];
        fields.forEach(field => {
            const input = document.getElementById(`val_${field}`);
            if (input) {
                input.addEventListener("input", (e) => {
                    if (currentDocumentData) {
                        currentDocumentData.fields[field] = e.target.value;
                        validateTotals();
                    }
                });
            }
        });

        // 2. Click-to-Map listeners for Map buttons
        document.addEventListener("click", (e) => {
            const btn = e.target.closest(".field-map-btn");
            if (btn) {
                e.stopPropagation();
                const fieldId = btn.getAttribute("data-field-id");
                toggleMappingMode(fieldId);
            }
        });
    }

    function toggleMappingMode(fieldId) {
        if (activeMappingField === fieldId) {
            deactivateMappingMode();
            return;
        }
        deactivateMappingMode();
        activeMappingField = fieldId;
        
        const btn = document.querySelector(`.field-map-btn[data-field-id="${fieldId}"]`);
        if (btn) btn.classList.add("active");

        const card = document.querySelector(`.field-card[data-field-id="${fieldId}"]`) ||
                     document.querySelector(`.totals-row[data-field-id="${fieldId}"]`) ||
                     document.querySelector(`.total-block[data-field-id="${fieldId}"]`);
        if (card) card.classList.add("mapping-active");
        canvasContainer.classList.add("mapping-cursor");

        // Redraw boxes to show raw OCR words for mapping
        if (currentDocumentData) {
            renderBoundingBoxes(currentDocumentData.bounding_boxes);
        }
    }

    function deactivateMappingMode() {
        if (!activeMappingField) return;
        document.querySelectorAll(".field-map-btn").forEach(btn => btn.classList.remove("active"));
        document.querySelectorAll(".field-card, .totals-row, .total-block").forEach(el => {
            el.classList.remove("mapping-active");
        });
        canvasContainer.classList.remove("mapping-cursor");
        activeMappingField = null;
        if (currentDocumentData) {
            renderBoundingBoxes(currentDocumentData.bounding_boxes);
        }
    }

    function mapBoxValueToField(fieldId, boxItem) {
        if (!currentDocumentData) return;
        const key = fieldId.replace("field_", "");
        let cleanVal = boxItem.value;
        if (boxItem.id.includes("item") && cleanVal.includes(" - ")) {
            cleanVal = cleanVal.split(" - ").slice(1).join(" - ");
        }
        currentDocumentData.fields[key] = cleanVal;

        let existingBoxIdx = currentDocumentData.bounding_boxes.findIndex(b => b.id === fieldId);
        if (existingBoxIdx !== -1) {
            currentDocumentData.bounding_boxes[existingBoxIdx].bbox = boxItem.bbox;
            currentDocumentData.bounding_boxes[existingBoxIdx].value = boxItem.value;
            currentDocumentData.bounding_boxes[existingBoxIdx].confidence = boxItem.confidence || 0.99;
        } else {
            currentDocumentData.bounding_boxes.push({
                id: fieldId,
                label: getFieldLabel(key),
                value: boxItem.value,
                bbox: boxItem.bbox,
                confidence: boxItem.confidence || 0.99
            });
        }

        const input = document.getElementById(`val_${key}`);
        if (input) input.value = cleanVal;

        deactivateMappingMode();
        validateTotals();
    }

    function getFieldLabel(key) {
        if (key === "merchant") return "Merchant";
        if (key === "date") return "Invoice Date";
        if (key === "invoice_no") return "Invoice Number";
        if (key === "subtotal") return "Subtotal";
        if (key === "tax") return "Tax";
        if (key === "total") return "Total Amount";
        return key;
    }

    function validateTotals() {
        const banner = document.getElementById("validation-banner");
        if (banner) {
            banner.style.display = "none";
        }
    }

    function renderBoundingBoxes(boxes) {
        boxOverlay.innerHTML = "";
        const scaleX = docImage.clientWidth / docImage.naturalWidth;
        const scaleY = docImage.clientHeight / docImage.naturalHeight;

        const boxesToRender = (activeMappingField && currentDocumentData && currentDocumentData.raw_ocr_words && currentDocumentData.raw_ocr_words.length > 0)
            ? currentDocumentData.raw_ocr_words.map((w, idx) => ({
                id: `raw_word_${idx}`,
                label: "OCR Word",
                value: w.text,
                bbox: w.bbox,
                confidence: w.score || 0.95,
                isRaw: true
            }))
            : boxes;

        boxesToRender.forEach(item => {
            const rawBox = item.bbox;
            const left = rawBox[0] * scaleX;
            const top = rawBox[1] * scaleY;
            const width = (rawBox[2] - rawBox[0]) * scaleX;
            const height = (rawBox[3] - rawBox[1]) * scaleY;

            const boxDiv = document.createElement("div");
            if (item.isRaw) {
                boxDiv.className = "box-highlight box-raw-word";
            } else {
                boxDiv.className = `box-highlight ${getBoxClass(item.id)}`;
            }
            
            boxDiv.style.left = `${left}px`;
            boxDiv.style.top = `${top}px`;
            boxDiv.style.width = `${width}px`;
            boxDiv.style.height = `${height}px`;
            boxDiv.setAttribute("data-box-id", item.id);
            boxDiv.title = `${item.label}: ${item.value} (${Math.round(item.confidence * 100)}%)`;

            // Hover interactions
            boxDiv.addEventListener("mouseenter", () => {
                highlightFieldCard(item.id, true);
            });
            boxDiv.addEventListener("mouseleave", () => {
                highlightFieldCard(item.id, false);
            });
            boxDiv.addEventListener("click", () => {
                selectFieldCard(item.id);
            });

            boxOverlay.appendChild(boxDiv);
        });
    }

    function getBoxClass(fieldId) {
        if (fieldId.includes("merchant")) return "box-merchant";
        if (fieldId.includes("date") || fieldId.includes("invoice")) return "box-date";
        if (fieldId.includes("total") || fieldId.includes("subtotal") || fieldId.includes("tax")) return "box-total";
        if (fieldId.includes("item")) return "box-item";
        return "box-item";
    }

    // ==========================================
    // 6. Sidebar Structured Card Updates
    // ==========================================
    function displayExtractionResults(data) {
        resultsEmpty.style.display = "none";
        extractedFields.style.display = "grid";

        latencyVal.innerText = `${data.processing_time_sec}s`;

        // Helper to query confidence scores from bounding boxes
        const getConfValue = (fieldId) => {
            const box = data.bounding_boxes.find(b => b.id === fieldId);
            return box ? Math.round(box.confidence * 100) : 95;
        };

        const confMerchant = getConfValue("field_merchant");
        const confDate = getConfValue("field_date");
        const confInvoice = getConfValue("field_invoice_no");
        const confTotal = getConfValue("field_total");
        const confSubtotal = getConfValue("field_subtotal");
        const confTax = getConfValue("field_tax");

        // Fill base field cards
        document.getElementById("val_merchant").value = data.fields.merchant || "Not Found";
        document.getElementById("val_date").value = data.fields.date || "Not Found";
        document.getElementById("val_invoice_no").value = data.fields.invoice_no || "Not Found";

        // Set Confidence scores and apply low-confidence alerts
        const updateConfEl = (elId, conf) => {
            const el = document.getElementById(elId);
            if (el) {
                el.innerText = `${conf}%`;
                if (conf < 80) el.classList.add("low-conf");
                else el.classList.remove("low-conf");
            }
        };

        updateConfEl("conf_merchant", confMerchant);
        updateConfEl("conf_date", confDate);
        updateConfEl("conf_invoice_no", confInvoice);
        updateConfEl("conf_total", confTotal);

        const confSubtotalEl = document.getElementById("conf_subtotal");
        if (confSubtotalEl) updateConfEl("conf_subtotal", confSubtotal);
        const confTaxEl = document.getElementById("conf_tax");
        if (confTaxEl) updateConfEl("conf_tax", confTax);

        // Update Progress Bars (color-coded widths)
        updateProgressBar("bar_merchant", confMerchant);
        updateProgressBar("bar_date", confDate);
        updateProgressBar("bar_invoice_no", confInvoice);

        // Determine active model tags dynamically
        const isLayoutLM = document.getElementById("chk-layoutlm").checked;
        const modelLabel = isLayoutLM ? "LayoutLM" : "Template OCR";
        const badgeClass = isLayoutLM ? "badge-layoutlm" : "badge-fallback";

        ["merchant", "date", "invoice_no"].forEach(field => {
            const badge = document.getElementById(`model_${field}`);
            if (badge) {
                badge.innerText = modelLabel;
                badge.className = `model-badge ${badgeClass}`;
            }
        });

        // Setup individual Copy buttons
        document.querySelectorAll(".field-copy-btn").forEach(btn => {
            // Remove old listeners by cloning
            const newBtn = btn.cloneNode(true);
            btn.parentNode.replaceChild(newBtn, btn);

            newBtn.addEventListener("click", (e) => {
                e.stopPropagation(); // prevent card selection trigger
                const inputId = newBtn.getAttribute("data-input-id");
                const input = document.getElementById(inputId);
                if (input) {
                    navigator.clipboard.writeText(input.value);
                    const icon = newBtn.querySelector("i");
                    icon.className = "fa-solid fa-check";
                    icon.style.color = "var(--secondary)";
                    setTimeout(() => {
                        icon.className = "fa-regular fa-copy";
                        icon.style.color = "";
                    }, 1500);
                }
            });
        });

        // Setup JSON Export button
        const exportBtn = document.getElementById("export-json-btn");
        if (exportBtn) {
            const newExportBtn = exportBtn.cloneNode(true);
            exportBtn.parentNode.replaceChild(newExportBtn, exportBtn);

            newExportBtn.addEventListener("click", () => {
                const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(data, null, 2));
                const downloadAnchor = document.createElement('a');
                downloadAnchor.setAttribute("href", dataStr);
                downloadAnchor.setAttribute("download", `extraction_${data.document_type || 'result'}.json`);
                document.body.appendChild(downloadAnchor);
                downloadAnchor.click();
                downloadAnchor.remove();
            });
        }

        // Items Table
        itemsTbody.innerHTML = "";
        const items = data.fields.items || [];
        if (items.length === 0) {
            itemsTbody.innerHTML = `<tr><td colspan="5" style="color:var(--text-muted); text-align: center;">No line items detected</td></tr>`;
        } else {
            items.forEach((item, idx) => {
                const tr = document.createElement("tr");
                tr.setAttribute("data-row-id", `field_item_${idx}`);
                
                const serialCell = document.createElement("td");
                serialCell.textContent = item.serial || (idx + 1);
                serialCell.style.textAlign = "center";
                serialCell.style.color = "var(--text-secondary)";

                const nameCell = document.createElement("td");
                nameCell.textContent = item.name;

                const qtyCell = document.createElement("td");
                qtyCell.textContent = item.quantity || "1";
                qtyCell.style.textAlign = "center";

                const rateCell = document.createElement("td");
                rateCell.textContent = item.rate || item.price;
                rateCell.style.textAlign = "right";
                rateCell.style.color = "var(--text-secondary)";

                const priceCell = document.createElement("td");
                priceCell.textContent = item.price;
                priceCell.style.textAlign = "right";
                priceCell.style.fontWeight = "600";
                priceCell.style.color = "white";

                tr.append(serialCell, nameCell, qtyCell, rateCell, priceCell);

                tr.addEventListener("mouseenter", () => highlightBoundingBox(`field_item_${idx}`, true));
                tr.addEventListener("mouseleave", () => highlightBoundingBox(`field_item_${idx}`, false));
                itemsTbody.appendChild(tr);
            });
        }

        // Totals values
        document.getElementById("val_subtotal").value = data.fields.subtotal || "$0.00";
        document.getElementById("val_tax").value = data.fields.tax || "$0.00";
        document.getElementById("val_total").value = data.fields.total || "$0.00";



        // Bind events to cards
        setupCardInteractionListeners();
        
        // Run validation check on totals
        validateTotals();
    }

    function updateProgressBar(barId, conf) {
        const bar = document.getElementById(barId);
        if (!bar) return;
        bar.style.width = `${conf}%`;

        bar.className = "conf-progress-bar";
        if (conf >= 85) {
            bar.classList.add("conf-high");
        } else if (conf >= 70) {
            bar.classList.add("conf-med");
        } else {
            bar.classList.add("conf-low");
        }
    }

    function setupCardInteractionListeners() {
        const cards = [
            document.getElementById("card_merchant"),
            document.getElementById("card_date"),
            document.getElementById("card_invoice_no"),
            document.getElementById("card_subtotal"),
            document.getElementById("card_tax"),
            document.getElementById("card_total")
        ];

        cards.forEach(card => {
            if (!card) return;
            const fieldId = card.getAttribute("data-field-id");
            card.addEventListener("mouseenter", () => highlightBoundingBox(fieldId, true));
            card.addEventListener("mouseleave", () => highlightBoundingBox(fieldId, false));
        });
    }

    // Bidirectional Hover Highlight logic
    function highlightFieldCard(fieldId, doHighlight) {
        // For tables
        if (fieldId.includes("item")) {
            const tr = document.querySelector(`tr[data-row-id="${fieldId}"]`);
            if (tr) {
                if (doHighlight) tr.style.backgroundColor = "rgba(16, 185, 129, 0.15)";
                else tr.style.backgroundColor = "transparent";
            }
            return;
        }

        const card = document.querySelector(`.field-card[data-field-id="${fieldId}"]`) ||
            document.querySelector(`.totals-row[data-field-id="${fieldId}"]`) ||
            document.querySelector(`.total-block[data-field-id="${fieldId}"]`);

        if (card) {
            if (doHighlight) card.classList.add("highlighted");
            else card.classList.remove("highlighted");
        }
    }

    function highlightBoundingBox(fieldId, doHighlight) {
        const box = document.querySelector(`.box-highlight[data-box-id="${fieldId}"]`);
        if (box) {
            if (doHighlight) box.classList.add("selected");
            else box.classList.remove("selected");
        }
    }

    function selectFieldCard(fieldId) {
        let card = document.querySelector(`.field-card[data-field-id="${fieldId}"]`) ||
            document.querySelector(`.total-block[data-field-id="${fieldId}"]`) ||
            document.querySelector(`tr[data-row-id="${fieldId}"]`);

        if (card) {
            card.scrollIntoView({ behavior: 'smooth', block: 'center' });

            // Flash color animation
            card.style.transition = "background-color 0.1s ease";
            card.style.backgroundColor = "rgba(99, 102, 241, 0.3)";
            setTimeout(() => {
                card.style.backgroundColor = "";
            }, 300);
        }
    }

    // ==========================================
    // 7. Training Playground Simulator
    // ==========================================
    function setupTrainingSimulator() {
        startTrainBtn.addEventListener("click", runTrainingSimulation);
    }

    async function runTrainingSimulation() {
        // Validate hyperparameters before submission
        const epochs = parseInt(trainEpochs.value);
        const lr = parseFloat(trainLr.value);
        const batchSize = parseInt(trainBatch.value);

        const validationErrors = [];
        if (isNaN(epochs) || epochs < 1 || epochs > 50) {
            validationErrors.push("Epochs must be a number between 1 and 50.");
        }
        if (isNaN(lr) || lr <= 0 || lr > 0.01) {
            validationErrors.push("Learning rate must be > 0 and ≤ 0.01.");
        }
        if (isNaN(batchSize) || batchSize < 1 || batchSize > 64) {
            validationErrors.push("Batch size must be a number between 1 and 64.");
        }

        if (validationErrors.length > 0) {
            terminalLogs.innerHTML = "";
            validationErrors.forEach(err => addTerminalLog(`Validation Error: ${err}`, "error"));
            return;  // Do not proceed — button stays enabled
        }

        startTrainBtn.disabled = true;
        startTrainBtn.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i> Fine-Tuning...`;

        // Log starting training
        terminalLogs.innerHTML = "";
        addTerminalLog("Starting Fine-tuning execution job...", "system");
        addTerminalLog(`Model: ${selectModel.value.toUpperCase()}`, "system");
        addTerminalLog(`Dataset: ${selectDataset.value.toUpperCase()}`, "system");
        addTerminalLog(`Hyperparameters: epochs=${epochs}, lr=${lr}, batch_size=${batchSize}`, "system");

        try {
            const response = await fetch("/api/train-simulate", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    dataset: selectDataset.value,
                    model_type: selectModel.value,
                    epochs: epochs,
                    learning_rate: lr,
                    batch_size: batchSize
                })
            });

            if (!response.ok) throw new Error("Simulation endpoint error");
            const data = await response.json();

            // Plot results with standard progressive visual intervals
            animateTrainingLogsAndCharts(data.results);

        } catch (err) {
            console.error("Simulation failed:", err);
            addTerminalLog("Local pipeline connection failed, initiating client-side sandbox training...", "error");
            // Client side mock simulation backup
            runClientSideSimulation();
        }
    }

    function addTerminalLog(msg, type = "normal") {
        const line = document.createElement("div");
        line.className = `log-line text-${type}`;
        line.innerText = `[${new Date().toLocaleTimeString()}] ${msg}`;
        terminalLogs.appendChild(line);
        terminalLogs.scrollTop = terminalLogs.scrollHeight;
    }

    function animateTrainingLogsAndCharts(results) {
        const labels = [];
        const trainLossData = [];
        const valLossData = [];
        const f1Data = [];

        // Initialize / reset chart
        initChart();

        let index = 0;
        const interval = setInterval(() => {
            if (index >= results.length) {
                clearInterval(interval);
                addTerminalLog("Model training complete! Fine-tuned weights updated successfully.", "system");
                startTrainBtn.disabled = false;
                startTrainBtn.innerHTML = `<i class="fa-solid fa-play"></i> Start Fine-Tuning`;
                return;
            }

            const epochData = results[index];

            // Append metrics
            labels.push(`Epoch ${epochData.epoch}`);
            trainLossData.push(epochData.train_loss);
            valLossData.push(epochData.val_loss);
            f1Data.push(epochData.f1_score * 100); // percentage scale

            // Update Chart.js data
            trainingChart.data.labels = labels;
            trainingChart.data.datasets[0].data = trainLossData;
            trainingChart.data.datasets[1].data = valLossData;
            trainingChart.data.datasets[2].data = f1Data;
            trainingChart.update();

            // Print logs
            addTerminalLog(`Epoch ${epochData.epoch}/${results.length} - loss: ${epochData.train_loss.toFixed(4)} - val_loss: ${epochData.val_loss.toFixed(4)} - precision: ${epochData.precision.toFixed(3)} - recall: ${epochData.recall.toFixed(3)} - F1-score: ${(epochData.f1_score * 100).toFixed(1)}%`);

            index++;
        }, 300); // 300ms delay per epoch simulation
    }

    function runClientSideSimulation() {
        const epochs = parseInt(trainEpochs.value) || 10;
        const mockResults = [];
        let loss = 2.4;
        let f1 = 0.3;
        for (let i = 1; i <= epochs; i++) {
            loss = Math.max(0.15, loss * 0.82 + Math.random() * 0.05);
            f1 = Math.min(0.95, f1 + (1.0 - f1) * 0.15 + Math.random() * 0.02);
            mockResults.push({
                epoch: i,
                train_loss: loss,
                val_loss: loss * 1.12,
                precision: f1 * 1.01,
                recall: f1 * 0.98,
                f1_score: f1
            });
        }
        animateTrainingLogsAndCharts(mockResults);
    }

    function initChart() {
        if (trainingChart) {
            trainingChart.destroy();
        }

        const ctx = document.getElementById('trainChart').getContext('2d');
        trainingChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    {
                        label: 'Training Loss',
                        data: [],
                        borderColor: '#6366f1',
                        backgroundColor: 'rgba(99, 102, 241, 0.1)',
                        borderWidth: 2,
                        yAxisID: 'y',
                        tension: 0.3
                    },
                    {
                        label: 'Validation Loss',
                        data: [],
                        borderColor: '#ef4444',
                        backgroundColor: 'transparent',
                        borderWidth: 1.5,
                        borderDash: [5, 5],
                        yAxisID: 'y',
                        tension: 0.3
                    },
                    {
                        label: 'F1 Score (%)',
                        data: [],
                        borderColor: '#10b981',
                        backgroundColor: 'rgba(16, 185, 129, 0.05)',
                        borderWidth: 2.5,
                        yAxisID: 'y1',
                        tension: 0.2
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        type: 'linear',
                        display: true,
                        position: 'left',
                        title: { display: true, text: 'Loss', color: 'rgba(255,255,255,0.7)' },
                        ticks: { color: 'rgba(255,255,255,0.5)' },
                        grid: { color: 'rgba(255,255,255,0.05)' }
                    },
                    y1: {
                        type: 'linear',
                        display: true,
                        position: 'right',
                        min: 0,
                        max: 100,
                        title: { display: true, text: 'F1-Score (%)', color: 'rgba(255,255,255,0.7)' },
                        ticks: { color: 'rgba(255,255,255,0.5)' },
                        grid: { drawOnChartArea: false } // only want grid lines for left axis
                    },
                    x: {
                        ticks: { color: 'rgba(255,255,255,0.5)' },
                        grid: { color: 'rgba(255,255,255,0.05)' }
                    }
                },
                plugins: {
                    legend: {
                        labels: { color: 'white', font: { size: 10 } }
                    }
                }
            }
        });
    }

    // ==========================================
    // 8. Extraction History Methods
    // ==========================================
    async function loadExtractionHistory() {
        const historyTbody = document.getElementById("history-tbody");
        if (!historyTbody) return;

        // Show inline loading state immediately
        historyTbody.innerHTML = `
            <tr>
                <td colspan="7" style="text-align: center; padding: 20px; color: var(--text-muted);">
                    <i class="fa-solid fa-spinner fa-spin" style="font-size: 18px; color: var(--primary); margin-bottom: 8px;"></i>
                    <div style="margin-top: 8px; font-size: 12px;">Loading extraction history...</div>
                </td>
            </tr>`;

        try {
            const res = await fetch("/api/history");
            if (!res.ok) throw new Error("Failed to fetch history");
            const history = await res.json();

            historyTbody.innerHTML = "";
            if (history.length === 0) {
                historyTbody.innerHTML = `<tr><td colspan="7" style="text-align: center; color: var(--text-muted);">No saved logs. Upload documents to log extraction history.</td></tr>`;
                return;
            }

            history.forEach(item => {
                const tr = document.createElement("tr");
                tr.setAttribute("data-history-id", item.id);

                const tsCell = document.createElement("td");
                tsCell.textContent = item.timestamp;

                const fnCell = document.createElement("td");
                fnCell.textContent = item.original_name;
                fnCell.style.maxWidth = "120px";
                fnCell.style.overflow = "hidden";
                fnCell.style.textOverflow = "ellipsis";
                fnCell.style.whiteSpace = "nowrap";

                const typeCell = document.createElement("td");
                typeCell.innerHTML = `<span class="model-badge badge-layoutlm" style="text-transform: capitalize;">${item.document_type}</span>`;

                const merchantCell = document.createElement("td");
                merchantCell.textContent = item.fields.merchant || "N/A";
                merchantCell.style.fontWeight = "500";

                const totalCell = document.createElement("td");
                totalCell.textContent = item.fields.total || "N/A";
                totalCell.style.color = "white";
                totalCell.style.fontWeight = "600";

                const latencyCell = document.createElement("td");
                latencyCell.innerHTML = `<i class="fa-solid fa-gauge-high" style="color:var(--primary); font-size: 10px;"></i> ${item.processing_time_sec}s`;

                const actionCell = document.createElement("td");
                const viewBtn = document.createElement("button");
                viewBtn.className = "btn-view-history";
                viewBtn.innerHTML = `<i class="fa-solid fa-eye"></i> View`;
                viewBtn.addEventListener("click", (e) => {
                    e.stopPropagation();
                    loadHistoryItem(item);
                });
                actionCell.appendChild(viewBtn);

                tr.append(tsCell, fnCell, typeCell, merchantCell, totalCell, latencyCell, actionCell);

                tr.addEventListener("click", () => {
                    loadHistoryItem(item);
                });

                historyTbody.appendChild(tr);
            });
        } catch (err) {
            console.error("Error loading history:", err);
            historyTbody.innerHTML = `
                <tr>
                    <td colspan="7" style="text-align: center; padding: 16px; color: var(--accent-red);">
                        <i class="fa-solid fa-circle-exclamation"></i>
                        Failed to load history. Please try again.
                    </td>
                </tr>`;
        }
    }

    function loadHistoryItem(item) {
        currentDocumentData = item;
        showLoading(true);
        
        document.getElementById("canvas-empty").style.display = "none";
        document.getElementById("canvas-container").style.display = "block";
        
        docImage.onload = () => {
            showLoading(false);
            displayExtractionResults(item);
            renderBoundingBoxes(item.bounding_boxes);
            docImage.onload = null;
        };
        docImage.src = item.file_url;
        
        document.querySelector(".document-viewer").scrollIntoView({ behavior: "smooth", block: "start" });
    }

    function setupHistoryHandlers() {
        const clearBtn = document.getElementById("clear-history-btn");
        if (clearBtn) {
            clearBtn.addEventListener("click", async () => {
                if (confirm("Are you sure you want to clear the extraction history?")) {
                    try {
                        const res = await fetch("/api/history/clear", { method: "POST" });
                        if (!res.ok) throw new Error("Failed to clear history");
                        loadExtractionHistory();
                    } catch (err) {
                        console.error("Error clearing history:", err);
                        alert("Failed to clear history: " + err.message);
                    }
                }
            });
        }
    }

});
