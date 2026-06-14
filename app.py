"""
Invoice AI System - Flask Backend
Render pe deploy hoga. Mistral AI use karta hai.
"""

import os
import json
import logging
import hashlib
import base64
import re
from datetime import datetime
from io import BytesIO
from typing import Dict, List, Any, Optional

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, origins=["https://invoice-ai-web-opal.vercel.app", "http://localhost:3000"], allow_headers=["Content-Type", "Authorization"], methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"])

# ── Mistral Client ────────────────────────────────────────────────────────────
from mistralai import Mistral

def get_mistral():
    key = os.environ.get("MISTRAL_API_KEY")
    if not key:
        raise ValueError("MISTRAL_API_KEY not set")
    return Mistral(api_key=key)

# ── Supabase Client ───────────────────────────────────────────────────────────
from supabase import create_client

def get_supabase():
    url = os.environ.get('SUPABASE_URL')
    key = os.environ.get('SUPABASE_SERVICE_KEY')
    if not url or not key:
        raise ValueError("SUPABASE_URL or SUPABASE_SERVICE_KEY not set")
    return create_client(url, key)

# ── OCR Helpers ───────────────────────────────────────────────────────────────
def extract_text_from_file(file_bytes: bytes, filename: str, mimetype: str) -> str:
    """Extract raw text from PDF or image using local OCR."""
    ext = filename.rsplit('.', 1)[-1].lower() if '.' in filename else ''
    text = ""

    try:
        if mimetype == 'application/pdf' or ext == 'pdf':
            import pdfplumber
            with pdfplumber.open(BytesIO(file_bytes)) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text() or ""
                    text += page_text + "\n"
        elif mimetype.startswith('image/') or ext in ('png', 'jpg', 'jpeg', 'tiff', 'bmp', 'webp'):
            import pytesseract
            from PIL import Image
            image = Image.open(BytesIO(file_bytes))
            text = pytesseract.image_to_string(image, lang='eng')
    except Exception as e:
        logger.warning(f"OCR extraction failed for {filename}: {e}")

    return text.strip()

# ── Mistral Extraction ────────────────────────────────────────────────────────
EXTRACTION_SYSTEM = """You are an expert invoice data extraction AI specializing in Indian GST invoices.
Extract all data exactly as shown. Return ONLY valid JSON, no markdown.

Indian GST rules:
- CGST + SGST = tax for intra-state; IGST for inter-state
- GSTIN: 15 chars (2 digit state + PAN 10 chars + 1 digit + Z + check digit)
- HSN codes: 4, 6, or 8 digits"""

EXTRACTION_PROMPT = """Analyze this invoice and extract all structured data. Return ONLY this JSON structure, no extra text:

{
  "vendor": {"name": null, "gstin": null, "address": null, "contact": null},
  "customer": {"name": null, "gstin": null, "address": null},
  "invoice": {"invoice_number": null, "invoice_date": null, "due_date": null, "po_number": null},
  "items": [
    {"description": null, "hsn_code": null, "quantity": null, "unit": null,
     "rate": null, "discount": null, "tax_percent": null,
     "cgst_amount": null, "sgst_amount": null, "igst_amount": null, "total_amount": null}
  ],
  "totals": {"subtotal": null, "cgst_total": null, "sgst_total": null,
              "igst_total": null, "tax_total": null, "grand_total": null, "currency": "INR"},
  "metadata": {"confidence_score": 0.8, "extraction_notes": ""}
}

Rules:
- Dates in YYYY-MM-DD format
- Amounts as numbers only (no currency symbols)
- Null for missing fields
- Extract ALL line items"""

def pdf_to_images_base64(file_bytes: bytes) -> List[str]:
    """Convert PDF pages to base64-encoded JPEG images."""
    try:
        from pdf2image import convert_from_bytes
        images = convert_from_bytes(file_bytes, dpi=200, fmt='jpeg')
        result = []
        for img in images[:5]:  # max 5 pages
            buf = BytesIO()
            img.save(buf, format='JPEG', quality=90)
            result.append(base64.b64encode(buf.getvalue()).decode('utf-8'))
        return result
    except Exception as e:
        logger.warning(f"pdf2image failed: {e}")
        return []


def extract_with_mistral(file_bytes: bytes, filename: str, mimetype: str, ocr_text: str = "") -> Dict:
    """Use Mistral pixtral to extract invoice data."""
    client = get_mistral()
    ext = filename.rsplit('.', 1)[-1].lower() if '.' in filename else ''

    content_parts = []

    # Prompt text
    prompt = EXTRACTION_PROMPT
    if ocr_text:
        prompt += f"\n\nOCR TEXT (use for reference):\n{ocr_text[:8000]}"

    content_parts.append({"type": "text", "text": prompt})

    # Attach file
    if mimetype == 'application/pdf' or ext == 'pdf':
        # Convert PDF pages to images (pixtral does not support document_url)
        page_images = pdf_to_images_base64(file_bytes)
        if page_images:
            for img_b64 in page_images:
                content_parts.append({
                    "type": "image_url",
                    "image_url": f"data:image/jpeg;base64,{img_b64}"
                })
        elif ocr_text:
            # Fallback: text-only if pdf2image failed
            content_parts = [{"type": "text", "text": f"{EXTRACTION_PROMPT}\n\nINVOICE TEXT:\n{ocr_text[:12000]}"}]
        else:
            raise ValueError("PDF conversion to images failed and no OCR text available")
    elif mimetype.startswith('image/') or ext in ('png', 'jpg', 'jpeg', 'tiff', 'bmp', 'webp'):
        base64_data = base64.b64encode(file_bytes).decode('utf-8')
        content_parts.append({
            "type": "image_url",
            "image_url": f"data:{mimetype};base64,{base64_data}"
        })
    elif ocr_text:
        # Text only
        content_parts = [{"type": "text", "text": f"{EXTRACTION_PROMPT}\n\nINVOICE TEXT:\n{ocr_text[:12000]}"}]

    response = client.chat.complete(
        model="pixtral-large-latest",
        messages=[
            {"role": "system", "content": EXTRACTION_SYSTEM},
            {"role": "user", "content": content_parts}
        ],
        temperature=0.1,
        max_tokens=4096,
    )

    raw = response.choices[0].message.content or ""

    # Parse JSON
    data = _parse_json(raw)
    return data

def _parse_json(text: str) -> Dict:
    """Extract JSON from response."""
    try:
        return json.loads(text)
    except:
        pass
    # Try markdown code block
    for pattern in [r'```json\s*([\s\S]*?)\s*```', r'```\s*([\s\S]*?)\s*```']:
        m = re.search(pattern, text)
        if m:
            try:
                return json.loads(m.group(1))
            except:
                pass
    # Try finding { ... }
    start = text.find('{')
    end = text.rfind('}') + 1
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end])
        except:
            pass
    return {"raw_text": text, "parse_error": True}

def _clean_gstin(val) -> Optional[str]:
    if not val:
        return None
    g = str(val).upper().strip().replace(' ', '')
    if len(g) == 15:
        return g
    m = re.search(r'\d{2}[A-Z]{5}\d{4}[A-Z]\d[Z]\d', g)
    return m.group() if m else (g or None)

def _parse_date(val) -> Optional[str]:
    if not val:
        return None
    from dateutil import parser as dateparser
    try:
        return dateparser.parse(str(val)).strftime('%Y-%m-%d')
    except:
        return None

def _parse_num(val) -> Optional[float]:
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val)
    s = re.sub(r'[₹$€,\s]', '', str(val)).replace('(', '-').replace(')', '')
    try:
        return float(s)
    except:
        return None

def clean_extracted(data: Dict) -> Dict:
    """Normalize extracted invoice data."""
    v = data.get('vendor', {}) or {}
    c = data.get('customer', {}) or {}
    inv = data.get('invoice', {}) or {}
    totals = data.get('totals', {}) or {}
    meta = data.get('metadata', {}) or {}

    items = []
    for item in (data.get('items') or []):
        items.append({
            'description': str(item.get('description') or '').strip() or None,
            'hsn_code': str(item.get('hsn_code') or '').strip() or None,
            'quantity': _parse_num(item.get('quantity')),
            'unit': str(item.get('unit') or '').strip() or None,
            'rate': _parse_num(item.get('rate')),
            'discount': _parse_num(item.get('discount')),
            'tax_percent': _parse_num(item.get('tax_percent')),
            'cgst_amount': _parse_num(item.get('cgst_amount')),
            'sgst_amount': _parse_num(item.get('sgst_amount')),
            'igst_amount': _parse_num(item.get('igst_amount')),
            'total_amount': _parse_num(item.get('total_amount')),
        })

    return {
        'vendor': {
            'name': str(v.get('name') or '').strip() or None,
            'gstin': _clean_gstin(v.get('gstin')),
            'address': str(v.get('address') or '').strip() or None,
            'contact': str(v.get('contact') or '').strip() or None,
        },
        'customer': {
            'name': str(c.get('name') or '').strip() or None,
            'gstin': _clean_gstin(c.get('gstin')),
            'address': str(c.get('address') or '').strip() or None,
        },
        'invoice': {
            'invoice_number': str(inv.get('invoice_number') or '').strip() or None,
            'invoice_date': _parse_date(inv.get('invoice_date')),
            'due_date': _parse_date(inv.get('due_date')),
            'po_number': str(inv.get('po_number') or '').strip() or None,
        },
        'items': items,
        'totals': {
            'subtotal': _parse_num(totals.get('subtotal')),
            'cgst_total': _parse_num(totals.get('cgst_total')),
            'sgst_total': _parse_num(totals.get('sgst_total')),
            'igst_total': _parse_num(totals.get('igst_total')),
            'tax_total': _parse_num(totals.get('tax_total')),
            'grand_total': _parse_num(totals.get('grand_total')),
            'currency': totals.get('currency', 'INR') or 'INR',
        },
        'metadata': {
            'confidence_score': float(meta.get('confidence_score') or 0.7),
            'extraction_notes': str(meta.get('extraction_notes') or ''),
        }
    }

# ── Reconciliation ────────────────────────────────────────────────────────────
def reconcile(data: Dict) -> Dict:
    """Validate invoice totals against line items."""
    items = data.get('items', [])
    totals = data.get('totals', {})
    TOL = 0.02

    calc_subtotal = sum(_parse_num(i.get('total_amount')) or 0 for i in items)
    calc_cgst = sum(_parse_num(i.get('cgst_amount')) or 0 for i in items)
    calc_sgst = sum(_parse_num(i.get('sgst_amount')) or 0 for i in items)
    calc_igst = sum(_parse_num(i.get('igst_amount')) or 0 for i in items)
    calc_tax = calc_cgst + calc_sgst + calc_igst
    calc_grand = calc_subtotal + calc_tax

    stated_grand = _parse_num(totals.get('grand_total')) or 0
    stated_tax = _parse_num(totals.get('tax_total')) or 0
    stated_subtotal = _parse_num(totals.get('subtotal')) or 0

    issues = []
    if stated_grand and abs(calc_grand - stated_grand) > TOL:
        issues.append({
            'field': 'grand_total',
            'severity': 'high',
            'message': f'Grand total mismatch: calculated {calc_grand:.2f} vs stated {stated_grand:.2f}'
        })
    if stated_tax and abs(calc_tax - stated_tax) > TOL:
        issues.append({
            'field': 'tax_total',
            'severity': 'medium',
            'message': f'Tax total mismatch: calculated {calc_tax:.2f} vs stated {stated_tax:.2f}'
        })
    if stated_subtotal and calc_subtotal and abs(calc_subtotal - stated_subtotal) > TOL:
        issues.append({
            'field': 'subtotal',
            'severity': 'medium',
            'message': f'Subtotal mismatch: calculated {calc_subtotal:.2f} vs stated {stated_subtotal:.2f}'
        })

    return {
        'is_valid': len([i for i in issues if i['severity'] == 'high']) == 0,
        'issues': issues,
        'calculated': {
            'subtotal': round(calc_subtotal, 2),
            'cgst_total': round(calc_cgst, 2),
            'sgst_total': round(calc_sgst, 2),
            'igst_total': round(calc_igst, 2),
            'tax_total': round(calc_tax, 2),
            'grand_total': round(calc_grand, 2),
        }
    }

# ── Duplicate Detection ───────────────────────────────────────────────────────
def check_duplicate(file_hash: str, invoice_number: str, vendor_name: str) -> Dict:
    """Check for duplicates in Supabase."""
    try:
        sb = get_supabase()
        # Hash check
        r = sb.table('invoices').select('id, invoice_number, vendor_name').eq('file_hash', file_hash).execute()
        if r.data:
            return {'is_duplicate': True, 'type': 'exact_file', 'match': r.data[0]}

        # Invoice number + vendor check
        if invoice_number and vendor_name:
            r = sb.table('invoices').select('id, invoice_number, vendor_name') \
                .eq('invoice_number', invoice_number).execute()
            for row in (r.data or []):
                if row.get('vendor_name') and vendor_name:
                    ratio = _similarity(row['vendor_name'], vendor_name)
                    if ratio > 0.85:
                        return {'is_duplicate': True, 'type': 'invoice_number', 'match': row}
    except Exception as e:
        logger.warning(f"Duplicate check failed: {e}")

    return {'is_duplicate': False}

def _similarity(a: str, b: str) -> float:
    from difflib import SequenceMatcher
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

# ── DB Save ───────────────────────────────────────────────────────────────────
def save_invoice(data: Dict, file_hash: str, filename: str,
                 confidence: float, reconciliation: Dict) -> Optional[str]:
    """Save invoice to Supabase and return invoice id."""
    try:
        sb = get_supabase()
        inv = data.get('invoice', {})
        vendor = data.get('vendor', {})
        customer = data.get('customer', {})
        totals = data.get('totals', {})

        row = {
            'file_hash': file_hash,
            'filename': filename,
            'invoice_number': inv.get('invoice_number'),
            'invoice_date': inv.get('invoice_date'),
            'due_date': inv.get('due_date'),
            'po_number': inv.get('po_number'),
            'vendor_name': vendor.get('name'),
            'vendor_gstin': vendor.get('gstin'),
            'vendor_address': vendor.get('address'),
            'customer_name': customer.get('name'),
            'customer_gstin': customer.get('gstin'),
            'customer_address': customer.get('address'),
            'subtotal': totals.get('subtotal'),
            'cgst_total': totals.get('cgst_total'),
            'sgst_total': totals.get('sgst_total'),
            'igst_total': totals.get('igst_total'),
            'tax_total': totals.get('tax_total'),
            'grand_total': totals.get('grand_total'),
            'currency': totals.get('currency', 'INR'),
            'confidence_score': confidence,
            'reconciliation_valid': reconciliation.get('is_valid', True),
            'reconciliation_issues': json.dumps(reconciliation.get('issues', [])),
            'status': 'pending_review' if reconciliation.get('issues') else 'processed',
            'raw_data': json.dumps(data),
        }

        result = sb.table('invoices').insert(row).execute()
        invoice_id = result.data[0]['id'] if result.data else None

        # Save line items
        if invoice_id and data.get('items'):
            items_rows = []
            for idx, item in enumerate(data['items']):
                items_rows.append({
                    'invoice_id': invoice_id,
                    'line_number': idx + 1,
                    'description': item.get('description'),
                    'hsn_code': item.get('hsn_code'),
                    'quantity': item.get('quantity'),
                    'unit': item.get('unit'),
                    'rate': item.get('rate'),
                    'discount': item.get('discount'),
                    'tax_percent': item.get('tax_percent'),
                    'cgst_amount': item.get('cgst_amount'),
                    'sgst_amount': item.get('sgst_amount'),
                    'igst_amount': item.get('igst_amount'),
                    'total_amount': item.get('total_amount'),
                })
            sb.table('invoice_items').insert(items_rows).execute()

        return invoice_id
    except Exception as e:
        logger.error(f"DB save failed: {e}")
        return None

# ═══════════════════════════════════════════════════════════════════════════════
# API ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

@app.route('/', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'service': 'Invoice AI Backend (Mistral AI)', 'version': '2.0'})

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'ok', 'timestamp': datetime.utcnow().isoformat()})

# ── POST /api/extract ─────────────────────────────────────────────────────────
@app.route('/api/extract', methods=['POST'])
def extract_invoice():
    """Extract invoice data from uploaded file."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded. Use field name: file'}), 400

    file = request.files['file']
    if not file.filename:
        return jsonify({'error': 'Empty filename'}), 400

    file_bytes = file.read()
    filename = file.filename
    mimetype = file.mimetype or 'application/octet-stream'

    # File hash for duplicate detection
    file_hash = hashlib.sha256(file_bytes).hexdigest()

    logger.info(f"[Extract] {filename} ({mimetype}, {len(file_bytes)} bytes)")

    try:
        # Step 1: Local OCR (fast pre-extraction)
        ocr_text = extract_text_from_file(file_bytes, filename, mimetype)
        logger.info(f"[OCR] Extracted {len(ocr_text)} chars")

        # Step 2: Duplicate check
        dup_result = check_duplicate(
            file_hash,
            None, None  # Will use full data after extraction
        )
        if dup_result['is_duplicate']:
            return jsonify({
                'success': False,
                'error': 'Duplicate invoice detected',
                'duplicate_type': dup_result.get('type'),
                'duplicate_match': dup_result.get('match'),
                'is_duplicate': True
            }), 409

        # Step 3: Mistral AI extraction
        raw_data = extract_with_mistral(file_bytes, filename, mimetype, ocr_text)

        if raw_data.get('parse_error'):
            return jsonify({'error': 'AI extraction failed to parse response', 'raw': raw_data.get('raw_text', '')}), 500

        # Step 4: Clean & normalize
        cleaned = clean_extracted(raw_data)

        # Step 5: Post-extraction duplicate check (invoice number)
        dup2 = check_duplicate(
            file_hash,
            cleaned['invoice'].get('invoice_number'),
            cleaned['vendor'].get('name')
        )
        if dup2['is_duplicate']:
            return jsonify({
                'success': False,
                'error': 'Duplicate invoice detected (same invoice number & vendor)',
                'duplicate_type': dup2.get('type'),
                'is_duplicate': True,
                'extracted_data': cleaned
            }), 409

        # Step 6: Reconciliation
        recon = reconcile(cleaned)
        cleaned['reconciliation'] = recon

        # Step 7: Save to DB
        invoice_id = save_invoice(cleaned, file_hash, filename,
                                   cleaned['metadata']['confidence_score'], recon)

        return jsonify({
            'success': True,
            'invoice_id': invoice_id,
            'data': cleaned,
            'is_duplicate': False,
            'filename': filename,
        })

    except Exception as e:
        logger.error(f"[Extract] Error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

# ── GET /api/invoices ─────────────────────────────────────────────────────────
@app.route('/api/invoices', methods=['GET'])
def list_invoices():
    """List invoices with filters."""
    try:
        sb = get_supabase()
        status = request.args.get('status')
        vendor = request.args.get('vendor')
        date_from = request.args.get('date_from')
        date_to = request.args.get('date_to')
        limit = int(request.args.get('limit', 50))
        offset = int(request.args.get('offset', 0))

        q = sb.table('invoices').select(
            'id, invoice_number, invoice_date, vendor_name, vendor_gstin, '
            'customer_name, grand_total, currency, status, confidence_score, '
            'reconciliation_valid, filename, created_at'
        ).order('created_at', desc=True).range(offset, offset + limit - 1)

        if status:
            q = q.eq('status', status)
        if vendor:
            q = q.ilike('vendor_name', f'%{vendor}%')
        if date_from:
            q = q.gte('invoice_date', date_from)
        if date_to:
            q = q.lte('invoice_date', date_to)

        result = q.execute()
        return jsonify({'invoices': result.data or [], 'count': len(result.data or [])})

    except Exception as e:
        logger.error(f"[List] {e}")
        return jsonify({'error': str(e)}), 500

# ── GET /api/invoices/:id ─────────────────────────────────────────────────────
@app.route('/api/invoices/<invoice_id>', methods=['GET'])
def get_invoice(invoice_id):
    """Get single invoice with line items."""
    try:
        sb = get_supabase()
        inv = sb.table('invoices').select('*').eq('id', invoice_id).single().execute()
        if not inv.data:
            return jsonify({'error': 'Invoice not found'}), 404

        items = sb.table('invoice_items').select('*').eq('invoice_id', invoice_id).order('line_number').execute()
        data = dict(inv.data)
        data['items'] = items.data or []
        return jsonify(data)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ── PATCH /api/invoices/:id ───────────────────────────────────────────────────
@app.route('/api/invoices/<invoice_id>', methods=['PATCH'])
def update_invoice(invoice_id):
    """Update invoice status or fields."""
    try:
        sb = get_supabase()
        body = request.get_json() or {}
        allowed = ['status', 'invoice_number', 'invoice_date', 'vendor_name',
                   'grand_total', 'notes']
        update_data = {k: v for k, v in body.items() if k in allowed}
        if not update_data:
            return jsonify({'error': 'No valid fields to update'}), 400

        sb.table('invoices').update(update_data).eq('id', invoice_id).execute()
        return jsonify({'success': True})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ── DELETE /api/invoices/:id ──────────────────────────────────────────────────
@app.route('/api/invoices/<invoice_id>', methods=['DELETE'])
def delete_invoice(invoice_id):
    try:
        sb = get_supabase()
        sb.table('invoice_items').delete().eq('invoice_id', invoice_id).execute()
        sb.table('invoices').delete().eq('id', invoice_id).execute()
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ── GET /api/dashboard ────────────────────────────────────────────────────────
@app.route('/api/dashboard', methods=['GET'])
def dashboard_stats():
    """Get dashboard statistics."""
    try:
        sb = get_supabase()

        # Total invoices & value
        all_inv = sb.table('invoices').select('id, grand_total, status, confidence_score, created_at').execute()
        rows = all_inv.data or []

        total_count = len(rows)
        total_value = sum(float(r.get('grand_total') or 0) for r in rows)
        avg_confidence = (sum(float(r.get('confidence_score') or 0) for r in rows) / total_count) if total_count else 0

        status_dist = {}
        for r in rows:
            s = r.get('status', 'unknown')
            status_dist[s] = status_dist.get(s, 0) + 1

        # Today
        today = datetime.utcnow().strftime('%Y-%m-%d')
        today_rows = [r for r in rows if (r.get('created_at') or '')[:10] == today]

        # This month
        this_month = datetime.utcnow().strftime('%Y-%m')
        month_rows = [r for r in rows if (r.get('created_at') or '')[:7] == this_month]

        # Top vendors
        vendor_counts = {}
        for r in sb.table('invoices').select('vendor_name, grand_total').execute().data or []:
            vn = r.get('vendor_name') or 'Unknown'
            if vn not in vendor_counts:
                vendor_counts[vn] = {'count': 0, 'total': 0}
            vendor_counts[vn]['count'] += 1
            vendor_counts[vn]['total'] += float(r.get('grand_total') or 0)

        top_vendors = sorted(
            [{'name': k, **v} for k, v in vendor_counts.items()],
            key=lambda x: x['count'], reverse=True
        )[:5]

        # Weekly trend (last 7 days)
        from collections import defaultdict
        trend = defaultdict(int)
        for r in rows:
            day = (r.get('created_at') or '')[:10]
            if day:
                trend[day] += 1
        weekly_trend = [{'date': k, 'count': v} for k, v in sorted(trend.items())[-7:]]

        return jsonify({
            'total_invoices': total_count,
            'total_value': round(total_value, 2),
            'avg_confidence': round(avg_confidence, 2),
            'today_count': len(today_rows),
            'month_count': len(month_rows),
            'month_value': round(sum(float(r.get('grand_total') or 0) for r in month_rows), 2),
            'status_breakdown': status_dist,
            'top_vendors': top_vendors,
            'weekly_trend': weekly_trend,
        })

    except Exception as e:
        logger.error(f"[Dashboard] {e}")
        return jsonify({'error': str(e)}), 500

# ── GET /api/reports/excel ────────────────────────────────────────────────────
@app.route('/api/reports/excel', methods=['GET'])
def export_excel():
    """Export invoices as Excel report."""
    try:
        import openpyxl
        from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
        from openpyxl.utils import get_column_letter

        sb = get_supabase()
        date_from = request.args.get('date_from')
        date_to = request.args.get('date_to')
        status = request.args.get('status')

        q = sb.table('invoices').select('*').order('invoice_date', desc=True)
        if date_from: q = q.gte('invoice_date', date_from)
        if date_to: q = q.lte('invoice_date', date_to)
        if status: q = q.eq('status', status)

        invoices = q.execute().data or []

        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "Invoice Report"

        # Header style
        header_fill = PatternFill(start_color="1F4E78", end_color="1F4E78", fill_type="solid")
        header_font = Font(color="FFFFFF", bold=True, size=11)
        center = Alignment(horizontal='center', vertical='center')

        headers = [
            'Invoice #', 'Date', 'Vendor Name', 'Vendor GSTIN',
            'Customer Name', 'Subtotal', 'CGST', 'SGST', 'IGST',
            'Tax Total', 'Grand Total', 'Currency', 'Status',
            'Confidence', 'Recon Valid', 'Filename', 'Created'
        ]

        for col, h in enumerate(headers, 1):
            cell = ws.cell(row=1, column=col, value=h)
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = center

        # Data rows
        alt_fill = PatternFill(start_color="D9E2F3", end_color="D9E2F3", fill_type="solid")
        for row_idx, inv in enumerate(invoices, 2):
            fill = alt_fill if row_idx % 2 == 0 else None
            row_data = [
                inv.get('invoice_number'), inv.get('invoice_date'),
                inv.get('vendor_name'), inv.get('vendor_gstin'),
                inv.get('customer_name'), inv.get('subtotal'),
                inv.get('cgst_total'), inv.get('sgst_total'), inv.get('igst_total'),
                inv.get('tax_total'), inv.get('grand_total'), inv.get('currency', 'INR'),
                inv.get('status'), inv.get('confidence_score'),
                'Yes' if inv.get('reconciliation_valid') else 'No',
                inv.get('filename'), (inv.get('created_at') or '')[:10]
            ]
            for col, val in enumerate(row_data, 1):
                cell = ws.cell(row=row_idx, column=col, value=val)
                if fill:
                    cell.fill = fill
                cell.alignment = Alignment(vertical='center')

        # Auto width
        for col in range(1, len(headers) + 1):
            max_len = max(
                len(str(ws.cell(row=r, column=col).value or ''))
                for r in range(1, len(invoices) + 2)
            )
            ws.column_dimensions[get_column_letter(col)].width = min(max_len + 4, 40)

        ws.freeze_panes = 'A2'

        # Summary sheet
        ws2 = wb.create_sheet("Summary")
        ws2.cell(1, 1, "Invoice AI System - Report Summary").font = Font(bold=True, size=14)
        ws2.cell(2, 1, f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
        ws2.cell(3, 1, f"Total Invoices: {len(invoices)}")
        ws2.cell(4, 1, f"Total Value: {sum(float(i.get('grand_total') or 0) for i in invoices):,.2f}")

        output = BytesIO()
        wb.save(output)
        output.seek(0)

        fname = f"invoice_report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.xlsx"
        return send_file(output, mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                         as_attachment=True, download_name=fname)

    except Exception as e:
        logger.error(f"[Excel] {e}")
        return jsonify({'error': str(e)}), 500

# ── GET /api/settings ─────────────────────────────────────────────────────────
@app.route('/api/settings', methods=['GET'])
def get_settings():
    try:
        sb = get_supabase()
        result = sb.table('settings').select('*').execute()
        settings = {r['key']: r['value'] for r in (result.data or [])}
        return jsonify(settings)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/settings', methods=['POST'])
def save_settings():
    try:
        sb = get_supabase()
        body = request.get_json() or {}
        for key, value in body.items():
            sb.table('settings').upsert({'key': key, 'value': str(value)}, on_conflict='key').execute()
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ── GET /api/vendors ──────────────────────────────────────────────────────────
@app.route('/api/vendors', methods=['GET'])
def list_vendors():
    try:
        sb = get_supabase()
        result = sb.table('invoices').select('vendor_name, vendor_gstin').execute()
        seen = set()
        vendors = []
        for r in (result.data or []):
            vn = r.get('vendor_name')
            if vn and vn not in seen:
                seen.add(vn)
                vendors.append({'name': vn, 'gstin': r.get('vendor_gstin')})
        return jsonify({'vendors': sorted(vendors, key=lambda x: x['name'])})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ── POST /api/reconcile ───────────────────────────────────────────────────────
@app.route('/api/reconcile/<invoice_id>', methods=['POST'])
def reconcile_invoice(invoice_id):
    """Re-run reconciliation on existing invoice."""
    try:
        sb = get_supabase()
        inv = sb.table('invoices').select('raw_data').eq('id', invoice_id).single().execute()
        if not inv.data:
            return jsonify({'error': 'Not found'}), 404

        raw = json.loads(inv.data.get('raw_data') or '{}')
        recon = reconcile(raw)

        sb.table('invoices').update({
            'reconciliation_valid': recon['is_valid'],
            'reconciliation_issues': json.dumps(recon['issues']),
            'status': 'pending_review' if recon['issues'] else 'processed'
        }).eq('id', invoice_id).execute()

        return jsonify(recon)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
