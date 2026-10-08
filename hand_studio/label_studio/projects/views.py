"""This file and its contents are licensed under the Apache License 2.0. Please see the included NOTICE for copyright information and LICENSE for a copy of the license.
"""
import logging

from django.contrib.auth.decorators import login_required
from django.shortcuts import render

from django.core.exceptions import PermissionDenied

logger = logging.getLogger(__name__)


@login_required
def project_list(request):
    return render(request, 'projects/list.html')


@login_required
def project_settings(request, pk, sub_path):
    if not (request.user.is_staff or request.user.is_superuser):
        raise PermissionDenied("Settings are restricted to administrators only.")
    return render(request, 'projects/settings.html')
