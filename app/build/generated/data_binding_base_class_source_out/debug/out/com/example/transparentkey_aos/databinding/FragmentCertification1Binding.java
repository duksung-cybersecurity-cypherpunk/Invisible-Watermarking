// Generated by view binder compiler. Do not edit!
package com.example.transparentkey_aos.databinding;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.constraintlayout.widget.ConstraintLayout;
import androidx.viewbinding.ViewBinding;
import androidx.viewbinding.ViewBindings;
import com.example.transparentkey_aos.R;
import java.lang.NullPointerException;
import java.lang.Override;
import java.lang.String;

public final class FragmentCertification1Binding implements ViewBinding {
  @NonNull
  private final ConstraintLayout rootView;

  @NonNull
  public final TextView cert1ErrorTv;

  @NonNull
  public final Button cert1ExtractBtn;

  @NonNull
  public final Button cert1GalleryBtn;

  @NonNull
  public final ImageView cert1Img1Iv;

  @NonNull
  public final ImageView certification1Back;

  @NonNull
  public final TextView textView;

  private FragmentCertification1Binding(@NonNull ConstraintLayout rootView,
      @NonNull TextView cert1ErrorTv, @NonNull Button cert1ExtractBtn,
      @NonNull Button cert1GalleryBtn, @NonNull ImageView cert1Img1Iv,
      @NonNull ImageView certification1Back, @NonNull TextView textView) {
    this.rootView = rootView;
    this.cert1ErrorTv = cert1ErrorTv;
    this.cert1ExtractBtn = cert1ExtractBtn;
    this.cert1GalleryBtn = cert1GalleryBtn;
    this.cert1Img1Iv = cert1Img1Iv;
    this.certification1Back = certification1Back;
    this.textView = textView;
  }

  @Override
  @NonNull
  public ConstraintLayout getRoot() {
    return rootView;
  }

  @NonNull
  public static FragmentCertification1Binding inflate(@NonNull LayoutInflater inflater) {
    return inflate(inflater, null, false);
  }

  @NonNull
  public static FragmentCertification1Binding inflate(@NonNull LayoutInflater inflater,
      @Nullable ViewGroup parent, boolean attachToParent) {
    View root = inflater.inflate(R.layout.fragment_certification1, parent, false);
    if (attachToParent) {
      parent.addView(root);
    }
    return bind(root);
  }

  @NonNull
  public static FragmentCertification1Binding bind(@NonNull View rootView) {
    // The body of this method is generated in a way you would not otherwise write.
    // This is done to optimize the compiled bytecode for size and performance.
    int id;
    missingId: {
      id = R.id.cert1_error_tv;
      TextView cert1ErrorTv = ViewBindings.findChildViewById(rootView, id);
      if (cert1ErrorTv == null) {
        break missingId;
      }

      id = R.id.cert1_extract_btn;
      Button cert1ExtractBtn = ViewBindings.findChildViewById(rootView, id);
      if (cert1ExtractBtn == null) {
        break missingId;
      }

      id = R.id.cert1_gallery_btn;
      Button cert1GalleryBtn = ViewBindings.findChildViewById(rootView, id);
      if (cert1GalleryBtn == null) {
        break missingId;
      }

      id = R.id.cert1_img1_iv;
      ImageView cert1Img1Iv = ViewBindings.findChildViewById(rootView, id);
      if (cert1Img1Iv == null) {
        break missingId;
      }

      id = R.id.certification1_back;
      ImageView certification1Back = ViewBindings.findChildViewById(rootView, id);
      if (certification1Back == null) {
        break missingId;
      }

      id = R.id.textView;
      TextView textView = ViewBindings.findChildViewById(rootView, id);
      if (textView == null) {
        break missingId;
      }

      return new FragmentCertification1Binding((ConstraintLayout) rootView, cert1ErrorTv,
          cert1ExtractBtn, cert1GalleryBtn, cert1Img1Iv, certification1Back, textView);
    }
    String missingId = rootView.getResources().getResourceName(id);
    throw new NullPointerException("Missing required view with ID: ".concat(missingId));
  }
}
