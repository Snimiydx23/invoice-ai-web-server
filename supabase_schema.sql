-- Invoice AI System - Supabase Schema
-- Supabase SQL Editor mein run karo

-- ── Invoices Table ──────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS invoices (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  file_hash TEXT UNIQUE,
  filename TEXT,
  invoice_number TEXT,
  invoice_date DATE,
  due_date DATE,
  po_number TEXT,
  vendor_name TEXT,
  vendor_gstin TEXT,
  vendor_address TEXT,
  customer_name TEXT,
  customer_gstin TEXT,
  customer_address TEXT,
  subtotal DECIMAL(15,2),
  cgst_total DECIMAL(15,2),
  sgst_total DECIMAL(15,2),
  igst_total DECIMAL(15,2),
  tax_total DECIMAL(15,2),
  grand_total DECIMAL(15,2),
  currency TEXT DEFAULT 'INR',
  confidence_score DECIMAL(4,3),
  reconciliation_valid BOOLEAN DEFAULT true,
  reconciliation_issues TEXT,
  status TEXT DEFAULT 'processed' CHECK (status IN ('processed', 'pending_review', 'approved', 'rejected', 'duplicate')),
  notes TEXT,
  raw_data TEXT,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- ── Invoice Items Table ──────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS invoice_items (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  invoice_id UUID REFERENCES invoices(id) ON DELETE CASCADE,
  line_number INTEGER,
  description TEXT,
  hsn_code TEXT,
  quantity DECIMAL(15,3),
  unit TEXT,
  rate DECIMAL(15,2),
  discount DECIMAL(15,2),
  tax_percent DECIMAL(6,2),
  cgst_amount DECIMAL(15,2),
  sgst_amount DECIMAL(15,2),
  igst_amount DECIMAL(15,2),
  total_amount DECIMAL(15,2),
  created_at TIMESTAMPTZ DEFAULT NOW()
);

-- ── Settings Table ───────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS settings (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  key TEXT UNIQUE NOT NULL,
  value TEXT,
  category TEXT DEFAULT 'general',
  updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- ── Default Settings ─────────────────────────────────────────────────────────
INSERT INTO settings (key, value, category) VALUES
  ('company_name', '', 'company'),
  ('company_gstin', '', 'company'),
  ('company_address', '', 'company'),
  ('theme', 'dark', 'ui'),
  ('confidence_threshold', '0.7', 'processing'),
  ('auto_reconcile', 'true', 'processing'),
  ('duplicate_check', 'true', 'processing')
ON CONFLICT (key) DO NOTHING;

-- ── Indexes ───────────────────────────────────────────────────────────────────
CREATE INDEX IF NOT EXISTS idx_invoices_invoice_number ON invoices(invoice_number);
CREATE INDEX IF NOT EXISTS idx_invoices_vendor_name ON invoices(vendor_name);
CREATE INDEX IF NOT EXISTS idx_invoices_invoice_date ON invoices(invoice_date);
CREATE INDEX IF NOT EXISTS idx_invoices_status ON invoices(status);
CREATE INDEX IF NOT EXISTS idx_invoices_file_hash ON invoices(file_hash);
CREATE INDEX IF NOT EXISTS idx_invoice_items_invoice_id ON invoice_items(invoice_id);

-- ── Auto-update updated_at ────────────────────────────────────────────────────
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = NOW();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER invoices_updated_at
  BEFORE UPDATE ON invoices
  FOR EACH ROW EXECUTE FUNCTION update_updated_at();
