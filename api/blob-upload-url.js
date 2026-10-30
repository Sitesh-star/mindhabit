import crypto from 'node:crypto';
import { issueSignedToken, presignUrl } from '@vercel/blob';

const MAX_BYTES = 25 * 1024 * 1024;
const DATASET_PATH = /^datasets\/[0-9a-f-]{36}\/[A-Za-z0-9_.-]+\.csv$/i;

export default async function handler(request, response) {
  if (request.method !== 'POST') {
    response.setHeader('Allow', 'POST');
    return response.status(405).json({ error: 'Method not allowed.' });
  }

  const { pathname, size, claim } = request.body || {};
  const secret = process.env.SECRET_KEY;
  if (!secret || !DATASET_PATH.test(pathname || '') ||
      !Number.isInteger(size) || size <= 0 || size > MAX_BYTES ||
      typeof claim !== 'string') {
    return response.status(400).json({ error: 'Invalid upload request.' });
  }

  const expected = crypto.createHmac('sha256', secret)
    .update(`${pathname}:${size}`).digest('hex');
  const claimBuffer = Buffer.from(claim);
  const expectedBuffer = Buffer.from(expected);
  if (claimBuffer.length !== expectedBuffer.length ||
      !crypto.timingSafeEqual(claimBuffer, expectedBuffer)) {
    return response.status(403).json({ error: 'Upload authorization is invalid.' });
  }

  try {
    const validUntil = Date.now() + 15 * 60 * 1000;
    const token = await issueSignedToken({
      pathname,
      operations: ['put'],
      validUntil,
      allowedContentTypes: ['text/csv'],
      maximumSizeInBytes: MAX_BYTES,
    });
    const result = await presignUrl(token, {
      pathname,
      operation: 'put',
      validUntil,
      allowedContentTypes: ['text/csv'],
      maximumSizeInBytes: MAX_BYTES,
    });
    return response.status(200).json({ uploadUrl: result.presignedUrl });
  } catch (error) {
    console.error('Blob upload authorization failed', error);
    return response.status(503).json({ error: 'Secure upload is temporarily unavailable.' });
  }
}
